"""Benchmarks page."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from matpes import MATPES_SRC

dash.register_page(__name__, path="/dataset", order=4)

INTRO_CONTENT = f"""
#### Introduction

Each MatPES dataset is provided as a gzipped file in the Javascript object notation (JSON) format. For example, the
`MatPES-PBE-2025.1.json.gz` file contains a list of structures with PES (energy, force, stresses) and associated
metadata. The [PBE]({MATPES_SRC}/MatPES-PBE-atoms.json.gz) and [r2SCAN]({MATPES_SRC}/MatPES-R2SCAN-atoms.json.gz)
atomic energies computed with the same  settings are also available. """

EXAMPLE_CONTENT = (
    """
#### Example document

The following is a commented version of a single entry in the `MatPES-PBE-2025.1.json.gz` file.

```json
{
    "builder_meta": { ... },  // Metadata used by MatPES developers only.

    "nsites": 2,  // Number of sites in the structure.
    "elements": ["Ti", "Y"],  // Elements present in the structure.
    "nelements": 2,  // Number of unique elements in the structure.

    "composition": { "Y": 1.0, "Ti": 1.0 },  // Elemental composition as a dictionary.
    "composition_reduced": { "Y": 1.0, "Ti": 1.0 },  // Reduced/normalized composition.

    "formula_pretty": "YTi",  // Readable chemical formula.
    "formula_anonymous": "AB",  // Anonymous formula representation.
    "chemsys": "Ti-Y",  // Chemical system association.

    "volume": 49.25681734779065,  // Structure volume in Å³.
    "density": 4.6108675489852535,  // Density in g/cm³.
    "density_atomic": 24.628408673895326,  // Atomic density.

    "symmetry": {  // Crystallographic symmetry information.
        "crystal_system": "Monoclinic",
        "symbol": "Pm",
        "number": 6,
        "point_group": "m",
        "symprec": 0.1,
        "angle_tolerance": 5.0,
        "version": "2.5.0"
    },

    "structure": { ... },  // Pymatgen serialized structure.

    "energy": -13.19442081,  // DFT energy in eV.

    "forces": [  // DFT-calculated forces on each atom (eV/Å).
        [0.43578007, -0.32456562, -0.38019986],
        [-0.43578007, 0.32456562, 0.38019986]
    ],

    "stress": [  // DFT-calculated stress tensor components (kilobar).
        -5.71186022, -9.34590239, 13.64346365,
        22.84178803, 23.6719352, 6.22290851
    ],

    "abs_forces": [  // Magnitude of DFT forces per atom (eV/Å).
        0.6631734649691654,
        0.6631734649691654
    ],

    "matpes_id": "matpes-20240214_999484_73",  // Unique MatPES ID for this structure.

    "bandgap": 0.0,  // DFT-calculated electronic band gap (eV).
    "functional": "PBE",  // DFT exchange-correlation functional.

    "formation_energy_per_atom": 0.5199284258333332,  // Formation energy per atom (eV).
    "cohesive_energy_per_atom": -4.266150985,  // Cohesive energy per atom (eV).

    "provenance": {  // Metadata describing dataset origin and simulation conditions.
        "original_mp_id": "mp-999484",  // Source ID from the Materials Project.
        "materials_project_version": "v2022.10.28",
        "md_ensemble": "NpT",  // Molecular dynamics ensemble type.
        "md_temperature": 300.0,  // MD simulation temperature (K).
        "md_pressure": 1.0,  // MD simulation pressure (atm).
        "md_step": 93,  // MD simulation step number.
        "mlip_name": "M3GNet-MP-2021.2.8-DIRECT"  // Machine learning potential used.
    }
}
```
""",
    f"""
#### Train-validation-test split

If you wish to reproduce the exact train:validation:test split used in the MatPES paper, you can download the split
indices for the [PBE]({MATPES_SRC}/MatPES-PBE-split.json.gz) and [r2SCAN]({MATPES_SRC}/MatPES-R2SCAN-split.json.gz).
You can then use the following code to split the dataset into train, validation, and test sets:
""",
    """
```python
from monty.serialization import loadfn
import json

pbe = loadfn("MatPES-PBE-2025.1.json.gz")
splits = loadfn("MatPES-PBE-split.json.gz")

train_set = []
valid_set = []
test_set = []

for i, d in enumerate(pbe):
    if i in splits["train"]:
        train_set.append(d)
    elif i in splits["valid"]:
        valid_set.append(d)
    else:
        test_set.append(d)

print(f"{len(train_set)}-{len(valid_set)}-{len(test_set)}")
# Output is 391240-21735-21737
```

""",
)

layout = dbc.Container([html.Div([dcc.Markdown(INTRO_CONTENT), dcc.Markdown(EXAMPLE_CONTENT)])])
