"""Tools for directly working with a MatPES style DB."""

from __future__ import annotations

import pandas as pd
from pymongo import MongoClient


class MatPESDB:
    """A MatPES DB object. This requires access to a MatPES style DB. Typically meant for developers."""

    FUNCTIONALS = ("PBE", "r2SCAN")

    def __init__(self, dbname="matpes"):
        """
        Args:
            dbname (str): The name of the MatPES DB.
        """
        client = MongoClient()
        self.db = client[dbname]

    def get_df(self, functional: str) -> pd.DataFrame:
        """
        Retrieve data for the given functional from the MongoDB database.

        Args:
            functional (str): The functional to query (e.g., 'PBE').

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        collection = self.db[functional]
        return pd.DataFrame(
            collection.find(
                {},
                projection=[
                    "elements",
                    "energy",
                    "chemsys",
                    "cohesive_energy_per_atom",
                    "formation_energy_per_atom",
                    "natoms",
                    "nelements",
                ],
            )
        )