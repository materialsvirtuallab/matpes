# Introduction

This notebook shows how to use pymatgen to generate VASP inputs for MatPES-compatible calculations. This is very useful if you intend to generate additional training data to fine-tine MatPES models.


```python
from __future__ import annotations

from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.sets import MatPESStaticSet
```


```python
example_structure = Structure.from_spacegroup(
    "Im-3m", Lattice.cubic(3), ["Li", "Li"], [[0, 0, 0], [0.5, 0.5, 0.5]]
)
```


```python
vis = MatPESStaticSet(example_structure, xc_functional="PBE")
print(vis.incar)
```

    ALGO = Normal
    EDIFF = 1e-05
    ENAUG = 1360
    ENCUT = 680.0
    GGA = Pe
    ISMEAR = 0
    ISPIN = 2
    KSPACING = 0.22
    LAECHG = True
    LASPH = True
    LCHARG = True
    LMAXMIX = 6
    LMIXTAU = True
    LORBIT = 11
    LREAL = False
    LWAVE = False
    MAGMOM = 4*0.6
    NELM = 200
    NSW = 0
    PREC = Accurate
    SIGMA = 0.05



Note the strict ENCUT and EDIFF used.


```python
# To write the input files to a directory, use the following line.
# vis.write_input("Li")
```

Similarly, the r2SCAN data can be generated using the following code:


```python
vis = MatPESStaticSet(example_structure, xc_functional="r2SCAN")
print(vis.incar)
```

    ALGO = All
    EDIFF = 1e-05
    ENAUG = 1360
    ENCUT = 680.0
    ISMEAR = 0
    ISPIN = 2
    KSPACING = 0.22
    LAECHG = True
    LASPH = True
    LCHARG = True
    LMAXMIX = 6
    LMIXTAU = True
    LORBIT = 11
    LREAL = False
    LWAVE = False
    MAGMOM = 4*0.6
    METAGGA = R2scan
    NELM = 200
    NSW = 0
    PREC = Accurate
    SIGMA = 0.05

