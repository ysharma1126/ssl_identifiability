# Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style [NeurIPS 2021]
Official code to reproduce the results and data presented in the paper [Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style](https://arxiv.org/abs/2106.04619).

<p align="center">
  <img src="https://github.com/ysharma1126/ssl_identifiability/blob/master/problem_formulation.png?raw=true" width="300" alt="Problem Formulation" />
</p>

## Numerical data
To train:
```bash
> python main_mlp.py --style-change-prob 0.75 --statistical-dependence --content-dependent-style
```
To evaluate:
```bash
> python main_mlp.py --style-change-prob 0.75 --statistical-dependence --content-dependent-style --evaluate
```

## Causal3DIdent Dataset
<p align="center">
  <img src="https://github.com/ysharma1126/ssl_identifiability/blob/master/causal_3dident.png?raw=true" alt="Causal3DIdent dataset example images" />
</p>

You can access the dataset [here](https://zenodo.org/record/4784282). The training and test datasets consists of 250000 and 25000 samples, respectively.

## High-dimensional images: Causal3DIdent
To train:
```bash
> python main_3dident.py --offline-dataset OFFLINE_DATASET --apply-random-crop --apply-color-distortion
```
To evaluate:
```bash
> python main_3dident.py --offline-dataset OFFLINE_DATASET --apply-random-crop --apply-color-distortion --evaluate
```

## BibTeX
```bibtex
@inproceedings{vonkugelgen2021self,
  title={Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style},
  author={von Kügelgen, Julius and Sharma, Yash and Gresele, Luigi and Brendel, Wieland and Schölkopf, Bernhard and Besserve, Michel and Locatello, Francesco},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
## Acknowledgements

This repository builds on the following [codebase](https://github.com/brendel-group/cl-ica). If you find the dataset/code provided here to be useful, I would recommend you to also cite the following, 
```bibtex
@article{zimmermann2021cl,
  author = {
    Zimmermann, Roland S. and
    Sharma, Yash and
    Schneider, Steffen and
    Bethge, Matthias and
    Brendel, Wieland
  },
  title = {
    Contrastive Learning Inverts
    the Data Generating Process
  },
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
    {ICML} 2021, 18-24 July 2021, Virtual Event},
  series = {Proceedings of Machine Learning Research},
  volume = {139},
  pages = {12979--12990},
  publisher = {{PMLR}},
  year = {2021},
  url = {http://proceedings.mlr.press/v139/zimmermann21a.html},
}
```
