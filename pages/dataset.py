"""Benchmarks page."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

dash.register_page(__name__, path="/dataset")

MARKDOWN_CONTENT = """
#### Introduction

Each MatPES dataset is provided as a pair of gzipped files in the Javascript object notation (JSON) format. For
example, the PBE dataset comprises `MatPES-PBE.json.gz` and `MatPES-atoms-PBE.json.gz`. The `MatPES-PBE.json.gz` file
contains a list of structures with PES (energy, force, stresses) and associated metadata. The `MatPES-atoms-PBE.json.gz`
provides reference atomic energies for each element, which is used to normalize the energy prior to training.

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
    "structure": {  # Pymatgen serialized structure.
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": 0.0,
        "lattice": {
            "matrix": [
                [
                    3.178955709039007,
                    0.0,
                    0.0
                ],
                [
                    0.9162579385309424,
                    6.0443487692359605,
                    0.0
                ],
                [
                    0.8505975627954393,
                    1.2610826193398135,
                    2.5634942881327865
                ]
            ],
            "pbc": [
                true,
                true,
                true
            ],
            "a": 3.178955709039007,
            "b": 6.11340172523328,
            "c": 2.9808301783100504,
            "alpha": 62.5448687562368,
            "beta": 73.41987469637797,
            "gamma": 81.3802049262522,
            "volume": 49.25681734779065
        },
        "properties": {},
        "sites": [
            {
                "species": [
                    {
                        "element": "Y",
                        "occu": 1
                    }
                ],
                "abc": [
                    0.8262975288031029,
                    0.1076795882636929,
                    0.0790466796845521
                ],
                "properties": {
                    "magmom": 0.147
                },
                "label": "Y",
                "xyz": [
                    2.792662437204551,
                    0.7505373806601974,
                    0.2026357118672113
                ]
            },
            {
                "species": [
                    {
                        "element": "Ti",
                        "occu": 1
                    }
                ],
                "abc": [
                    0.4313331046725963,
                    0.6086949234199679,
                    0.9632165015661229
                ],
                "properties": {
                    "magmom": 1.071
                },
                "label": "Ti",
                "xyz": [
                    2.748219999999993,
                    4.893859999999998,
                    2.469200000000001
                ]
            }
        ]
    },
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
