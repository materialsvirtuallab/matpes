[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matpes)](https://github.com/materialsvirtuallab/matpes/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)

#### Introduction

Welcome to MatPES, a foundational potential energy surface (PES) dataset for materials. This is a collaboration between
the [Materials Virtual Lab] and the [Materials Project].

Machine learning interatomic potentials (MLIPs) have revolutionized the field of computational materials science.
MLIPs use ML to reproduce the PES (energies, forces, and stresses) of a collection of atoms, typically computed
using an ab initio method such as density functional theory (DFT).
This enables the simulation of materials at much larger length and longer time scales at near-ab initio accuracy.

One of the most exciting developments in the past few years is the emergence of universal MLIPs (uMLIPs, aka materials
foundational models), with near-complete coverage of the periodic table of elements. Examples include [M3GNet],
[CHGNet], [MACE], to name a few. uMLIPs have broad applications, including materials discovery and the
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
2. **Diversity.** The structures in MatPES are robustly sampled from 300K MD simulations using the original [M3GNet]
   uMLIP. MatPES uses a modified version of DImensionality-Reduced Encoded Clusters with sTratified ([DIRECT])
   sampling to ensure comprehensive coverage of structures and local environments.
3. **Quality.** MatPES contains not only data computed using the PBE functional, but also the revised regularized SCAN
   (r2SCAN) meta-GGA functional. The r2SCAN functional recovers all 17 exact constraints presently known for
   meta-GGA functionals and has shown good transferable accuracy across diverse bonding and chemistries.

#### How to use MatPES

##### Download

You can download the entire MatPES dataset at [MatPES.ai](https://matpes.ai).
(note: links are not functional until publication).

##### MatPES tool

We have also provided a simple library to work with MatPES. Using this tool, you can download the MatPES dataset
and extract subsets of the data, e.g., by elements or chemical system.

Install the latest version of the `matpes` library using pip.

```shell
pip install matpes
```

The `matpes` tool comes with a command-line interface for working with the matpes dataset.

```shell
# Download the PBE dataset to the current directory
matpes download pbe

# You should see a MatPES-PBE-20240214.json.gz file in your directory.

# Extract all entries in the Fe-O chemical system
matpes data -i MatPES-PBE-20240214.json.gz --chemsys Fe-O -o Fe-O.json.gz
```

The `matpes.db` module also provides functionality to create your own MongoDB database with the MatPES downloaded data,
which is extremely useful if you are going to be working with the data (e.g., querying, adding entries, etc.) a lot.

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
[M3GNet]: http://dx.doi.org/10.1038/s43588-022-00349-3
[CHGNet]: http://doi.org/10.1038/s42256-023-00716-3
[MACE]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html
[DIRECT]: https//doi.org/10.1038/s41524-024-01227-4
