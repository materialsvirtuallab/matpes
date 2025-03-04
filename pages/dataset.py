"""Benchmarks page."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

dash.register_page(__name__, path="/dataset")

MARKDOWN_CONTENT = """
#### Introduction

Each MatPES dataset is provided as a gzipped file in the Javascript object notation (JSON) format. For example, the
`MatPES-PBE-20240214.json.gz` file contains a list of structures with PES (energy, force, stresses) and associated
metadata. The [PBE](https://mavrl-web.s3.us-east-1.amazonaws.com/matpes/MatPES-PBE-atoms.json.gz) and
[r2SCAN](https://mavrl-web.s3.us-east-1.amazonaws.com/matpes/MatPES-R2SCAN-atoms.json.gz) atomic energies computed
with the same  settings are also available, though you will probably not need them unless in special situations.

#### Example document

The following is a single document in the `MatPES-PBE.json.gz` file. Comments are provided for each field.

```json
{
    "builder_meta": {...},  # This metadata field is used by the MatPES developers only.
    "nsites": 2,  # Number of sites in the structure.
    "elements": ["Ti", "Y"],  # The elements in the structure.
    "nelements": 2,  # Number of elements for this structure.
    "composition": {"Y": 1.0, "Ti": 1.0},  # The composition as a dict.
    "composition_reduced": {"Y": 1.0, "Ti": 1.0},  # The reduced/normalized composition as a dict.
    "formula_pretty": "YTi",  # An easy to read string formula.
    "formula_anonymous": "AB",  # An anonymous formula.
    "chemsys": "Ti-Y",  # The chemical system the structure is associated with.
    "volume": 49.25681734779065,  # The volume of the structure.
    "density": 4.6108675489852535,  # The density of the structure.
    "density_atomic": 24.628408673895326,  # The atomic density of the structure.
    "symmetry": {
        "crystal_system": "Monoclinic",
        "symbol": "Pm",
        "number": 6,
        "point_group": "m",
        "symprec": 0.1,
        "angle_tolerance": 5.0,
        "version": "2.5.0"
    },
    "structure": {...},  # Pymatgen serialized structure.
    "energy": -13.19442081,  # The DFT energy per atom.
    "forces": [  # The DFT forces on each atom.
        [
            0.43578007,
            -0.32456562,
            -0.38019986
        ],
        [
            -0.43578007,
            0.32456562,
            0.38019986
        ]
    ],
    "stress": [  # The DFT stress.
        -5.71186022,
        -9.34590239,
        13.64346365,
        22.84178803,
        23.6719352,
        6.22290851
    ],
    "abs_forces": [  # The absolute value of the DFT forces on each atom.
        0.6631734649691654,
        0.6631734649691654
    ],
    "matpes_id": "matpes-20240214_999484_73",  # A unique id associated with each structure.
    "bandgap": 0.0,  # The DFT band gap.
    "functional": "PBE",  # The DFT functional.
    "formation_energy_per_atom": 0.5199284258333332,  # The DFT formation energy per atom.
    "cohesive_energy_per_atom": -4.266150985,  # The DFT cohesive energy per atom.
    "provenance": {  # Important metadata on how the structure was obtained.
        "original_mp_id": "mp-999484",
        "materials_project_version": "v2022.10.28",
        "md_ensemble": "NpT",
        "md_temperature": 300.0,
        "md_pressure": 1.0,
        "md_step": 93,
        "mlip_name": "M3GNet-MP-2021.2.8-DIRECT"
    }
}
```
"""

layout = dbc.Container([html.Div([dcc.Markdown(MARKDOWN_CONTENT)])])
