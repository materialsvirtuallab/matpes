[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matpes)](https://github.com/materialsvirtuallab/matpes/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/matpes/workflows/Testing/badge.svg)](https://github.com/materialsvirtuallab/matpes/workflows/Testing/badge.svg)

### About

Welcome to MatPES. MatPES is a foundational potential energy surface (PES) dataset for materials.

### Background

Machine learning interatomic potentials (MLIPs) have revolutionized the field of computational materials science.
MLIPs use ML to reproduce the PES (energies, forces, and stresses) of a collection of atoms, typically computed
using an ab initio method such as density functional theory (DFT).
This enables the simulation of materials at much larger length and longer time scales at near-ab initio accuracy.

One of the most exiciting developments in the past few years is the emergence of universal MLIPs (uMLIPs, aka materials
foundational models), with near-complete coverage of the periodic table of elements. Examples of such uMLIPs include
M3GNet,[^1] CHGNet,[^2] MACE,[^3] to name a few. uMLIPs have broad applications, including materials discovery and the
prediction of PES-derived properties such as elastic constants, phonon dispersion, etc.

### Goals

The aims of MatPES are three-fold:

1. **Accuracy.** The data in MatPES was computed using static DFT calculations with stringent converegence criteria.
   Please refer to the `MatPESStaticSet` in [pymatgen] for details.
2. **Diversity.** The structures in MatPES are robustly sampled from 300K MD simulations using the original M3GNet
   uMLIP.[^1] MatPES uses a modified version of DImensionality-Reduced Encoded Clusters with sTratified (DIRECT)
   sampling to ensure comprehensive coverage of structures and local environments.[^4]
3. **Quality.** MatPES contains not only data computed using the standard Perdew-Burke-Ernzerhof (PBE) generalized
   gradient approximation (GGA) functional, but also the revised regularized strongly constrained and appropriately
   normed (r2SCAN) meta-GGA functional. The r2SCAN functional recovers all 17 exact constraints presently known for
   meta-GGA functionals and has shown good transferable accuracy across diverse bonding and chemistries.

### How to use MatPES

You can download the entire MatPES dataset using the following links
(# Note that these links are defunct for now until the publication):

- [PBE](http://materialsvirtuallab.org)
- [R2SCAN](http://materialsvirtuallab.org)

The [MatPES Explorer](explorer) provides a statistical visualization of the dataset. We have also provided a simple tool
to extract subsets of the data, e.g., by elements or chemical system.

### Citing MatPES

If you use MatPES, please cite the following work:

```txt
Aaron Kaplan, Runze Liu, Ji Qi, Tsz Wai Ko, Bowen Deng, Gerbrand Ceder, Kristin A. Persson, Shyue Ping Ong.
A foundational potential energy surface dataset for materials. Submitted.
```

[pymatgen]: http://pymatgen.org

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
