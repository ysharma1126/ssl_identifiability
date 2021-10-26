import numpy as np
import torch
import argparse
import losses
import spaces
import disentanglement_utils
import invertible_network_utils
import torch.nn.functional as F
import random
import os
import latent_spaces
import encoders
from sklearn.preprocessing import StandardScaler
import string

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

def get_exp_name(args, parser, blacklist=['evaluate', 'num_train_batches', 'num_eval_batches', 'evaluate_iter']):
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
    parser.add_argument("--content-n", type=int, default=5)
    parser.add_argument("--style-n", type=int, default=5)
    parser.add_argument("--style-change-prob", type=float, default=1.0)
    parser.add_argument("--statistical-dependence", action='store_true')
    parser.add_argument("--content-dependent-style", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--num-train-batches", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--num-eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--c-param", type=float, default=1.0)
    parser.add_argument("--m-param", type=float, default=1.0)
    parser.add_argument("--n-mixing-layer", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--load-f", default=None)
    parser.add_argument("--load-g", default=None)
    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--n-log-steps", type=int, default=250)
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
        args.load_f = os.path.join(args.model_dir, get_exp_name(args, parser),'unsup_f.pth')
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
    loss = losses.LpSimCLRLoss()
    latent_spaces_list = []
    for i in range(2):
        content_condition = (i == 0)
        space = spaces.NRealSpace(args.content_n if content_condition else args.style_n)
        eta = torch.zeros(args.content_n if content_condition else args.style_n)
        sample_marginal = lambda space, size, device=device: space.normal(
            None, args.m_param, size, device, 
            statistical_dependence=args.statistical_dependence
        )
        if content_condition:
            sample_conditional = lambda space, z, size, device=device: z
        else:
            sample_conditional = lambda space, z, size, device=device: space.normal(
                z, args.c_param, size, device, 
                change_prob=args.style_change_prob, 
                statistical_dependence=args.statistical_dependence
            )
        latent_spaces_list.append(latent_spaces.LatentSpace(
            space=space,
            sample_marginal=sample_marginal,
            sample_conditional=sample_conditional,
        ))
    latent_space = latent_spaces.ProductLatentSpace(spaces=latent_spaces_list, 
                            content_dependent_style=args.content_dependent_style)
    def sample_marginal_and_conditional(size, device=device):
        z = latent_space.sample_marginal(size=size, device=device)
        z3 = latent_space.sample_marginal(size=size, device=device)
        z_tilde = latent_space.sample_conditional(z, size=size, device=device)

        return z, z_tilde, z3

    g = invertible_network_utils.construct_invertible_mlp(
        n=args.content_n + args.style_n,
        n_layers=args.n_mixing_layer,
        cond_thresh_ratio=0.001,
        n_iter_cond_thresh=25000,
    )
    g = g.to(device)

    if args.load_g is not None:
        g.load_state_dict(torch.load(args.load_g, map_location=device))

    for p in g.parameters():
        p.requires_grad = False

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

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(g.state_dict(), os.path.join(args.save_dir, "g.pth"))

    def train_step(data, loss, optimizer):
        optimizer.zero_grad()
        z1, z2_con_z1, z3 = data
        z1 = z1.to(device)
        z2_con_z1 = z2_con_z1.to(device)
        z3 = z3.to(device)
        z1_rec = h(z1)
        z2_con_z1_rec = h(z2_con_z1)
        z3_rec = h(z3)
        total_loss_value, _, losses_value = loss(
            z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec
        )
        total_loss_value.backward()
        optimizer.step()
        return total_loss_value.item(), unpack_item_list(losses_value)
    f = encoders.get_mlp(
        n_in=args.content_n + args.style_n,
        n_out=args.content_n,
        layers=[
            (args.content_n + args.style_n) * 10,
            (args.content_n + args.style_n) * 50,
            (args.content_n + args.style_n) * 50,
            (args.content_n + args.style_n) * 50,
            (args.content_n + args.style_n) * 50,
            (args.content_n + args.style_n) * 10,
        ],
    )
    f = f.to(device)

    if args.load_f is not None:
        f.load_state_dict(torch.load(args.load_f, map_location=device))

    print("f: ", f)
    optimizer = torch.optim.Adam(f.parameters(), lr=args.lr)
    h = lambda z: f(g(z))

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
            data = sample_marginal_and_conditional(size=args.batch_size)
            total_loss_value, losses_value = train_step(
                data, loss=loss, optimizer=optimizer
            )
            total_loss_values.append(total_loss_value)
            individual_losses_values.append(losses_value)
        if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
            content_linear_scores = []
            style_linear_scores = []
            content_nonlinear_scores = []
            style_nonlinear_scores = []
            if args.evaluate:
                training_z = []
                training_hz = []
                for i in range(args.num_train_batches):
                    z_disentanglement = latent_space.sample_marginal(4096)
                    hz_disentanglement = h(z_disentanglement)
                    training_z.append(z_disentanglement)
                    training_hz.append(hz_disentanglement)
                training_z = torch.cat(training_z)
                training_hz = torch.cat(training_hz)
                scaler_hz = StandardScaler()
                hz = scaler_hz.fit_transform(training_hz.detach().cpu().numpy())
                content_scaler_z = StandardScaler()
                content_z = content_scaler_z.fit_transform(training_z[:,
                                                   :args.content_n].detach().cpu().numpy())
                style_scaler_z = StandardScaler()
                style_z = style_scaler_z.fit_transform(training_z[:, 
                                                   args.content_n:].detach().cpu().numpy())
                content_n_model = disentanglement_utils.nonlinear_disentanglement(
                        content_z, hz, train_mode=True
                    )
                style_n_model = disentanglement_utils.nonlinear_disentanglement(
                        style_z, hz, train_mode=True
                    )
                content_l_model = disentanglement_utils.linear_disentanglement(
                        content_z, hz, train_mode=True
                    )
                style_l_model = disentanglement_utils.linear_disentanglement(
                        style_z, hz, train_mode=True
                    )
            for i in range(args.num_eval_batches):
                z_disentanglement = latent_space.sample_marginal(4096)
                hz_disentanglement = h(z_disentanglement)
                content_z = z_disentanglement[:, :args.content_n]
                style_z = z_disentanglement[:, args.content_n:]
                if args.evaluate:
                    hz = scaler_hz.transform(hz_disentanglement.detach().cpu().numpy())
                    content_z = content_scaler_z.transform(content_z.detach().cpu().numpy())
                    style_z = style_scaler_z.transform(style_z.detach().cpu().numpy())
                    (
                        content_linear_disentanglement_score,
                        _,
                    ), _ = disentanglement_utils.linear_disentanglement(
                        content_z, hz, mode="r2", model=content_l_model
                    )
                    content_linear_scores.append(content_linear_disentanglement_score)
                    (
                        style_linear_disentanglement_score,
                        _,
                    ), _ = disentanglement_utils.linear_disentanglement(
                        style_z, hz, mode="r2", model=style_l_model
                    )
                    style_linear_scores.append(style_linear_disentanglement_score)
                    (
                        content_nonlinear_disentanglement_score,
                        _,
                    ), _ = disentanglement_utils.nonlinear_disentanglement(
                        content_z, hz, mode="r2", model=content_n_model, 
                    )
                    content_nonlinear_scores.append(content_nonlinear_disentanglement_score)
                    (
                        style_nonlinear_disentanglement_score,
                        _,
                    ), _ = disentanglement_utils.nonlinear_disentanglement(
                        style_z, hz, mode="r2", model=style_n_model,
                    )
                    style_nonlinear_scores.append(style_nonlinear_disentanglement_score)
                else:
                    (
                        content_linear_disentanglement_score,
                        _,
                    ), _ = disentanglement_utils.linear_disentanglement(
                        content_z, hz_disentanglement, mode="r2", train_test_split=True,
                    )
                    content_linear_scores.append(content_linear_disentanglement_score)
                    (
                        style_linear_disentanglement_score,
                        _,
                    ), _ = disentanglement_utils.linear_disentanglement(
                        style_z, hz_disentanglement, mode="r2", train_test_split=True,
                    )
                    style_linear_scores.append(style_linear_disentanglement_score)
            print(
                "content linear mean: {} std: {}".format(
                    np.mean(content_linear_scores), np.std(content_linear_scores)
                )
            )
            print(
                "style linear mean: {} std: {}".format(
                    np.mean(style_linear_scores), np.std(style_linear_scores)
                )
            )
            if args.evaluate:
                print(
                    "content nonlinear mean: {} std: {}".format(
                        np.mean(content_nonlinear_scores, axis=0), 
                        np.std(content_nonlinear_scores, 
                                                                          axis=0)
                    )
                )
                print(
                    "style nonlinear mean: {} std: {}".format(
                        np.mean(style_nonlinear_scores, axis=0), np.std(style_nonlinear_scores, 
                                                                        axis=0)
                    )
                )
            if not args.evaluate and (global_step % args.n_log_steps == 1 or global_step == args.n_steps):
                print(
                    f"Step: {global_step} \t",
                    f"Loss: {total_loss_value:.4f} \t",
                    f"<Loss>: {np.mean(np.array(total_loss_values[-args.n_log_steps:])):.4f} \t",
                )
            if args.save_dir:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                torch.save(
                    f.state_dict(),
                    os.path.join(
                        args.save_dir, "{}_f.pth".format("unsup")
                    ),
                )
        global_step += 1


if __name__ == "__main__":
    main()