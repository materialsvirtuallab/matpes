[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matpes)](https://github.com/materialsvirtuallab/matpes/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)

#### Welcome

Welcome to MatPES, a foundational potential energy surface (PES) dataset for materials. This is a collaboration between
the [Materials Virtual Lab] and the [Materials Project].

Machine learning interatomic potentials (MLIPs) have revolutionized the field of computational materials science.
MLIPs use ML to reproduce the PES (energies, forces, and stresses) of a collection of atoms, typically computed
using an ab initio method such as density functional theory (DFT).
This enables the simulation of materials at much larger length and longer time scales at near-ab initio accuracy.

One of the most exciting developments in the past few years is the emergence of universal MLIPs (uMLIPs, aka materials
foundational models), with near-complete coverage of the periodic table of elements. Examples include M3GNet,[^1]
CHGNet,[^2] MACE,[^3] to name a few. uMLIPs have broad applications, including materials discovery and the
prediction of PES-derived properties such as elastic constants, phonon dispersion, etc.

However, most current uMLIPs were trained on DFT relaxation calculations from the [Materials Project].
This dataset, referred to as `MPF` or `MPTraj` in the literature, suffer from several issues:

1. The energies, forces, and stresses are not converged to the accuracies necessary to train a high quality MLIP.
2. Most of the structures are near-equilibrium, with very little coverage of non-equilibrium local environments.
3. The calculations were performed using the common Perdew-Burke-Ernzerhof (PBE) generalized gradient approximation
   (GGA) functional, even though improved functionals with better performance across diverse chemistries and bonding
   such as the strongly constrained and appropriately normed (SCAN) meta-GGA functional already exists.

MatPES is a continuing effort to address these limitations comprehensively.

#### Goals

The aims of MatPES are three-fold:

1. **Accuracy.** The data in MatPES was computed using static DFT calculations with stringent converegence criteria.
   Please refer to the `MatPESStaticSet` in [pymatgen] for details.
2. **Diversity.** The structures in MatPES are robustly sampled from 300K MD simulations using the original M3GNet
   uMLIP.[^1] MatPES uses a modified version of DImensionality-Reduced Encoded Clusters with sTratified (DIRECT)
   sampling to ensure comprehensive coverage of structures and local environments.[^4]
3. **Quality.** MatPES contains not only data computed using the PBE functional, but also the revised regularized SCAN
   (r2SCAN) meta-GGA functional. The r2SCAN functional recovers all 17 exact constraints presently known for
   meta-GGA functionals and has shown good transferable accuracy across diverse bonding and chemistries.

#### How to use MatPES

##### Download

You can download the entire MatPES dataset at [MatPES.ai](https://matpes.ai).
(note: links are not functional until the publication).

We have also provided a simple tool to extract subsets of the data, e.g., by elements or chemical system.

##### Exploring

The [MatPES Explorer](explorer) provides a statistical visualization of the dataset.

##### Pre-trained uMLIPs

Together with MatPES, we have released a set of MatPES uMLIPs in various architectures (M3GNet, CHGNet, TensorNet) in
the [MatGL] package. This is by far the easiest way to get started with using MatPES.

##### Notebooks

We have provided a series of Jupyter notebooks demonstrating how to load the MatPES dataset, train a model and perform
finetuning.

#### Citing MatPES

If you use MatPES, please cite the following work:

```txt
Aaron Kaplan, Runze Liu, Ji Qi, Tsz Wai Ko, Bowen Deng, Gerbrand Ceder, Kristin A. Persson, Shyue Ping Ong.
A foundational potential energy surface dataset for materials. Submitted.
```

[Materials Virtual Lab]: http://materialsvirtuallab.org
[pymatgen]: https://pymatgen.org
[Materials Project]: https://materialsproject.org
[MatGL]: https://matgl.ai

[^1]: Chen, C.; Ong, S. P. A Universal Graph Deep Learning Interatomic Potential for the Periodic Table. Nat Comput
      Sci 2022, 2 (11), 718-728. DOI: [10.1038/s43588-022-00349-3](http://dx.doi.org/10.1038/s43588-022-00349-3).
[^2]: Deng, B.; Zhong, P.; Jun, K.; Riebesell, J.; Han, K.; Bartel, C. J.; Ceder, G. CHGNet as a Pretrained Universal
      Neural Network Potential for Charge-Informed Atomistic Modelling. Nat Mach Intell 2023, 5 (9), 1031-1041.
      DOI: [10.1038/s42256-023-00716-3](http://doi.org/10.1038/s42256-023-00716-3).
[^3]: Batatia, I.; Kovacs, D. P.; Simm, G.; Ortner, C.; Csanyi, G. MACE: Higher Order Equivariant Message Passing
      Neural Networks for Fast and Accurate Force Fields. Advances in Neural Information Processing Systems 2022, 35,
      11423-11436.
[^4]: Qi, J.; Ko, T. W.; Wood, B. C.; Pham, T. A.; Ong, S. P. Robust Training of Machine Learning Interatomic
      Potentials with Dimensionality Reduction and Stratified Sampling. npj Computational Materials 2024, 10 (43), 1-11.
      DOI: [10.1038/s41524-024-01227-4](https//doi.org/10.1038/s41524-024-01227-4).
