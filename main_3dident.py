import numpy as np
import torch
import argparse
import disentanglement_utils
import torch.nn.functional as F
import random
import os
import latent_spaces
from sklearn.preprocessing import StandardScaler
import string
from torchvision import models
from datasets.clevr_dataset import CausalDataset
from infinite_iterator import InfiniteIterator
import faiss
from torchvision import transforms
from PIL import Image

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
else:
    device = "cpu"

print("device:", device)

def valid_str(v):
    if hasattr(v, '__name__'):
        return valid_str(v.__name__)
    if isinstance(v, tuple) or isinstance(v, list):
        return '-'.join([valid_str(x) for x in v])
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = ''.join(c if c in valid_chars else '-' for c in str_v)
    return str_v

def get_exp_name(args, parser, blacklist=['evaluate',  
                                          'num_train_batches', 'num_eval_batches',
                                          'offline_dataset', 'evaluate_iter', 'n_eval_samples']):
    exp_name = ''
    for x in vars(args):
        if getattr(args, x) != parser.get_default(x) and x not in blacklist:
            if isinstance(getattr(args, x),bool):
                exp_name += ('_' + x) if getattr(args, x) else ''
            else:
                exp_name += '_' + x + valid_str(getattr(args, x))
    return exp_name.lstrip('_')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change-all-hues", action='store_true')
    parser.add_argument("--change-all-positions", action='store_true')
    parser.add_argument("--change-all-rotations", action='store_true')
    parser.add_argument("--offline-dataset", type=str, default="")
    parser.add_argument("--encoder", default="rn18", choices=("rn18", "rn50", "rn101", "rn151",
                                                              "ccrn18", "ccrn50", "ccrn101", "ccrn152"))
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--faiss-omp-threads", type=int, default=16)
    parser.add_argument("--workers", default=32, type=int)
    parser.add_argument("--apply-rotation", action='store_true')
    parser.add_argument("--apply-random-crop", action="store_true")
    parser.add_argument("--random-crop-type", default="small", type=str, choices=("small", "large"))
    parser.add_argument("--apply-color-distortion", action="store_true")
    parser.add_argument("--n-eval-samples", default=2048, type=int)
    parser.add_argument("--save-every", default=10000, type=int)
    parser.add_argument("--evaluate-iter", type=int, default=20000)
    parser.add_argument("--content-n", type=int, default=8)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--num-train-batches", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--num-eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--c-param", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--load-f", default=None)
    parser.add_argument("--load-g", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-log-steps", type=int, default=500)
    parser.add_argument("--n-steps", type=int, default=100001)
    parser.add_argument("--resume-training", action="store_true")
    args = parser.parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    return args, parser


def main():
    args, parser = parse_args()
    if not args.evaluate:
        args.save_dir = os.path.join(args.model_dir, get_exp_name(args, parser))
    else:
        args.load_f = os.path.join(args.model_dir, '{}.iteration_{}'.format(get_exp_name(args, parser), 
                                                                                args.evaluate_iter))
        args.n_steps = 1
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    global device
    if args.no_cuda:
        device = "cpu"
        print("Using cpu")

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    use_augmentations = args.apply_random_crop or args.apply_color_distortion or args.apply_rotation
    latent_space = None
    from torchvision import transforms
    faiss.omp_set_num_threads(args.faiss_omp_threads)
    mean_per_channel = [0.4327, 0.2689, 0.2839]
    std_per_channel = [0.1201, 0.1457, 0.1082]
    transform_list = []
    if args.apply_random_crop:
        transform_list += [transforms.RandomResizedCrop(224, 
                                    scale=(0.08 if args.random_crop_type == "small" else 0.8, 1.0), 
                                                        interpolation=Image.BICUBIC), 
                           transforms.RandomHorizontalFlip()]
    if args.apply_color_distortion:
        def ColourDistortion(s=1.0):
            # s is the strength of color distortion.
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
            return color_distort

        transform_list += [ColourDistortion()]
    transform_test_list = [transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_per_channel,
            std=std_per_channel
        )]
    transform_list += transform_test_list
    dataset_kwargs = dict(transform=transforms.Compose(transform_list))
    latent_dimensions_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset_kwargs["latent_dimensions_to_use"] = latent_dimensions_to_use
    if not args.evaluate:
        train_dataset = CausalDataset(classes=np.arange(7), 
                                      root='{}/trainset'.format(args.offline_dataset), 
                                      biaugment=True,
                                    use_augmentations=use_augmentations,
                                    change_all_positions=args.change_all_positions,
                                    change_all_hues=args.change_all_hues,
                                    change_all_rotations=args.change_all_rotations,
                                      apply_rotation=args.apply_rotation,
                                      **dataset_kwargs)
    dataset_kwargs['transform'] = transforms.Compose(transform_test_list)
    test_dataset = CausalDataset(classes=np.arange(7), 
                                 root='{}/testset'.format(args.offline_dataset), 
                                  biaugment=False, 
                                  **dataset_kwargs)
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                           num_workers=args.workers,
                                           pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                              num_workers=args.workers,
                                               pin_memory=True, shuffle=True)

    def unpack_item_list(lst):
        if isinstance(lst, tuple):
            lst = list(lst)
        result_list = []
        for it in lst:
            if isinstance(it, (tuple, list)):
                result_list.append(unpack_item_list(it))
            else:
                result_list.append(it.item())
        return result_list
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()

    def train_step(data, optimizer):
        optimizer.zero_grad()
        object_class, (z1, z2), (x1, x2) = data
        z1_rec = f(x1)
        z2_rec = f(x2)
        sim11 = sim_metric(z1_rec.unsqueeze(-2), z1_rec.unsqueeze(-3)) / args.tau
        sim22 = sim_metric(z2_rec.unsqueeze(-2), z2_rec.unsqueeze(-3)) / args.tau
        sim12 = sim_metric(z1_rec.unsqueeze(-2), z2_rec.unsqueeze(-3)) / args.tau
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        total_loss_value = criterion(raw_scores, targets)
        losses_value = [total_loss_value]
        total_loss_value.backward()
        optimizer.step()
        return total_loss_value.item(), unpack_item_list(losses_value)
    
    base_encoder_class = {
        "rn18": models.resnet18,
        "rn50": models.resnet50,
        "rn101": models.resnet101,
        "rn152": models.resnet152,
    }[args.encoder]
    n_latents = 10
    encoder = base_encoder_class(False, num_classes=n_latents * 10)
    projection = torch.nn.Sequential(*[torch.nn.LeakyReLU(), torch.nn.Linear(n_latents * 10, args.content_n + 1)])
    f = torch.nn.Sequential(*[encoder, projection])
    f = torch.nn.DataParallel(f)
    f = f.to(device)

    if args.load_f is not None:
        f.load_state_dict(torch.load(args.load_f, map_location=device))

    print("f: ", f)
    optimizer = torch.optim.Adam(f.parameters(), lr=args.lr)
    if not args.evaluate:
        train_iterator = InfiniteIterator(train_loader)
    test_iterator = InfiniteIterator(test_loader)
    if (
        "total_loss_values" in locals() and not args.resume_training
    ) or "total_loss_values" not in locals():
        individual_losses_values = []
        total_loss_values = []

    global_step = len(total_loss_values) + 1
    last_save_at_step = 0
    while (
        global_step <= args.n_steps
    ):
        if not args.evaluate:
            data = next(train_iterator)
            total_loss_value, losses_value = train_step(
                data, optimizer=optimizer
            )

            total_loss_values.append(total_loss_value)
            individual_losses_values.append(losses_value)
        if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
            linear_scores = []
            interior_scores = []
            nonlinear_scores = []
            noninterior_scores = []
            if args.evaluate:
                training_z = []
                training_hz = []
                training_objz = []
                training_intz = []
                for i in range(args.num_train_batches):
                    z_disentanglement, hz_disentanglement = [], []
                    intz_dis = []
                    objz_dis = []
                    with torch.no_grad():
                        for batch_idx in range(args.n_eval_samples // args.batch_size):
                            test_data = next(test_iterator)
                            batch_obj_dis, batch_z_dis, batch_x_dis = test_data
                            objz_dis.append(batch_obj_dis)
                            encoding = encoder(batch_x_dis.to(device)).detach().to(batch_z_dis.device)
                            intz_dis.append(encoding)
                            batch_h_z_ = projection(intz_dis[-1].to(device))
                            z_disentanglement.append(batch_z_dis)
                            hz_disentanglement.append(batch_h_z_)
                    z_disentanglement = torch.cat(z_disentanglement, 0)
                    hz_disentanglement = torch.cat(hz_disentanglement, 0)
                    obj_disentanglement = torch.cat(objz_dis, 0)
                    intz_disentanglement = torch.cat(intz_dis, 0)
                    training_z.append(z_disentanglement)
                    training_hz.append(hz_disentanglement)
                    training_intz.append(intz_disentanglement)
                    training_objz.append(obj_disentanglement)
                training_z = torch.cat(training_z)
                training_hz = torch.cat(training_hz)
                training_intz = torch.cat(training_intz)
                training_objz = torch.cat(training_objz)
                scaler_hz = StandardScaler()
                hz = scaler_hz.fit_transform(training_hz.detach().cpu().numpy())
                scaler_intz = StandardScaler()
                intz = scaler_intz.fit_transform(training_intz.detach().cpu().numpy())
                scaler_zs = []
                n_models = []
                l_models = []
                for i in range(training_z.size(-1)):
                    scaler_z = StandardScaler()
                    standardized_z = scaler_z.fit_transform(torch.reshape(training_z[:,i],
                                                                  (-1,1)).detach().cpu().numpy())
                    scaler_zs.append(scaler_z)
                    n_models.append(disentanglement_utils.nonlinear_disentanglement(
                        standardized_z, hz, train_mode=True
                    ))
                    l_models.append(disentanglement_utils.linear_disentanglement(
                        standardized_z, hz, train_mode=True
                    ))
                log_model = disentanglement_utils.linear_disentanglement(training_objz, hz, 
                                                                         train_mode=True, 
                                                                         mode="accuracy")
                int_n_models = []
                int_l_models = []
                for i in range(training_z.size(-1)):
                    scaler_z = StandardScaler()
                    standardized_z = scaler_zs[i].transform(torch.reshape(training_z[:,i],
                                                                   (-1,1)).detach().cpu().numpy())
                    int_n_models.append(disentanglement_utils.nonlinear_disentanglement(
                        standardized_z, intz, train_mode=True
                    ))
                    int_l_models.append(disentanglement_utils.linear_disentanglement(
                        standardized_z, intz, train_mode=True
                    ))
                int_log_model = disentanglement_utils.linear_disentanglement(training_objz, intz, 
                                                                         train_mode=True, 
                                                                         mode="accuracy")
            for i in range(args.num_eval_batches):
                z_disentanglement, hz_disentanglement = [], []
                obj_dis = []
                int_dis = []
                with torch.no_grad():
                    for batch_idx in range(args.n_eval_samples // args.batch_size):
                        test_data = next(test_iterator)
                        batch_obj_dis, batch_z_dis, batch_x_dis = test_data
                        obj_dis.append(batch_obj_dis)
                        int_dis.append(encoder(batch_x_dis.to(device)).detach().to(batch_z_dis.device))
                        batch_h_z_dis = projection(int_dis[-1].to(device))
                        z_disentanglement.append(batch_z_dis)
                        hz_disentanglement.append(batch_h_z_dis)
                z_disentanglement = torch.cat(z_disentanglement, 0)
                hz_disentanglement = torch.cat(hz_disentanglement, 0)
                obj_dis = torch.cat(obj_dis, 0)
                int_dis = torch.cat(int_dis, 0)
                if args.evaluate:
                    hz = scaler_hz.transform(hz_disentanglement.detach().cpu().numpy())
                    intz = scaler_intz.transform(int_dis.detach().cpu().numpy())
                    lin_scores = []
                    nonlin_scores = []
                    for i in range(z_disentanglement.size(-1)):
                        scaled_zi = scaler_zs[i].transform(torch.reshape(z_disentanglement[:,
                                                                             i], 
                                                                   (-1,1)).detach().cpu().numpy())
                        (
                            linear_disentanglement_score,
                            _,
                        ), _ = disentanglement_utils.linear_disentanglement(
                            scaled_zi, hz, 
                            mode="r2", model=l_models[i]
                        )
                        lin_scores.append(linear_disentanglement_score)
                        (
                            nonlinear_disentanglement_score,
                            _,
                        ), _ = disentanglement_utils.nonlinear_disentanglement(
                            scaled_zi, hz, 
                            mode="r2", model=n_models[i]
                        )
                        nonlin_scores.append(nonlinear_disentanglement_score)
                    (
                        log_score,
                        _,
                    ), _ = disentanglement_utils.linear_disentanglement(obj_dis, 
                                                                                      hz, 
                                                                              mode="accuracy", 
                                                                            model=log_model)
                    lin_scores.insert(0, log_score)
                    nonlin_scores.insert(0, lin_scores[0])
                    int_lin_scores = []
                    int_nonlin_scores = []
                    for i in range(z_disentanglement.size(-1)):
                        scaled_zi = scaler_zs[i].transform(torch.reshape(z_disentanglement[:,
                                                                             i],
                                                               (-1,1)).detach().cpu().numpy())
                        (
                            linear_disentanglement_score,
                            _,
                        ), _ = disentanglement_utils.linear_disentanglement(
                            scaled_zi, intz, 
                            mode="r2", model=int_l_models[i]
                        )
                        int_lin_scores.append(linear_disentanglement_score)
                        (
                            nonlinear_disentanglement_score,
                            _,
                        ), _ = disentanglement_utils.nonlinear_disentanglement(
                            scaled_zi, intz, 
                            mode="r2", model=int_n_models[i]
                        )
                        int_nonlin_scores.append(nonlinear_disentanglement_score)
                    (
                        log_score,
                        _,
                    ), _ = disentanglement_utils.linear_disentanglement(obj_dis, 
                                                                                      intz, 
                                                                              mode="accuracy", 
                                                                            model=int_log_model)
                    int_lin_scores.insert(0, log_score)
                    int_nonlin_scores.insert(0, int_lin_scores[0])
                    interior_scores.append(int_lin_scores)
                    noninterior_scores.append(int_nonlin_scores)
                    linear_scores.append(lin_scores)
                    nonlinear_scores.append(nonlin_scores)
                else:
                    lin_scores = []
                    int_scores = []
                    for i in range(z_disentanglement.size(-1)):
                        (
                            linear_disentanglement_score,
                            _,
                        ), _ = disentanglement_utils.linear_disentanglement(
                            z_disentanglement[:,i], hz_disentanglement, mode="r2", 
                            train_test_split=True,
                        )
                        lin_scores.append(linear_disentanglement_score)
                    (cls_sc, _), _ = disentanglement_utils.linear_disentanglement(obj_dis, 
                                                                           hz_disentanglement,
                                                                              mode="accuracy", 
                                                                        train_test_split=True)
                    lin_scores.append(cls_sc)
                    for i in range(z_disentanglement.size(-1)):
                        (
                            linear_disentanglement_score,
                            _,
                        ), _ = disentanglement_utils.linear_disentanglement(
                            z_disentanglement[:,i], int_dis, mode="r2", 
                            train_test_split=True,
                        )
                        int_scores.append(linear_disentanglement_score)
                    (cls_sc, _), _ = disentanglement_utils.linear_disentanglement(obj_dis, 
                                                                         int_dis,
                                                                              mode="accuracy", 
                                                                        train_test_split=True)
                    int_scores.append(cls_sc)
                    interior_scores.append(int_scores)
                    linear_scores.append(lin_scores)
            if args.evaluate:
                print('{} {} {} {} {} {} {} {} {} {} {}'.format(*[np.mean(np.array(linear_scores)[:,i]) for i in range(11)]))
                print('{} {} {} {} {} {} {} {} {} {} {}'.format(*[np.mean(np.array(nonlinear_scores)[:,i]) for i in range(11)]))
                print('{} {} {} {} {} {} {} {} {} {} {}'.format(*[np.mean(np.array(interior_scores)[:,i]) for i in range(11)]))
                print('{} {} {} {} {} {} {} {} {} {} {}'.format(*[np.mean(np.array(noninterior_scores)[:,i]) for i in range(11)]))
            else:
                for i in range(z_disentanglement.size(-1)):
                    print(
                        "linear mean: {} std: {}".format(
                            np.mean(np.array(linear_scores)[:,i]), 
                            np.std(np.array(linear_scores)[:,i])
                        )
                    )
                print(
                    "logistic mean: {} std: {}".format(
                        np.mean(np.array(linear_scores)[:,-1]), 
                        np.std(np.array(linear_scores)[:,-1])
                    )
                )
                for i in range(z_disentanglement.size(-1)):
                    print(
                        "int linear mean: {} std: {}".format(
                            np.mean(np.array(interior_scores)[:,i]), 
                            np.std(np.array(interior_scores)[:,i])
                        )
                    )
                print(
                    "int logistic mean: {} std: {}".format(
                        np.mean(np.array(interior_scores)[:,-1]), 
                        np.std(np.array(interior_scores)[:,-1])
                    )
                )
            if not args.evaluate and (global_step % args.n_log_steps == 1 or global_step == args.n_steps):
                print(
                    f"Step: {global_step} \t",
                    f"Loss: {total_loss_value:.4f} \t",
                    f"<Loss>: {np.mean(np.array(total_loss_values[-args.n_log_steps:])):.4f} \t",
                )
            if args.save_every is not None:
                if global_step // args.save_every != last_save_at_step // args.save_every:
                    last_save_at_step = global_step
                    model_path = args.save_dir + f".iteration_{global_step}"
                    torch.save(f.state_dict(), model_path)
                    torch.save(f.state_dict(), args.save_dir)
        global_step += 1
       


if __name__ == "__main__":
    main()
