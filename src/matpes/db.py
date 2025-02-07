"""Tools for directly working with a MatPES style DB."""

from __future__ import annotations

import os

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
        client = MongoClient(
            host=os.environ.get("MATPES_HOST", "127.0.0.1"),
            username=os.environ.get("MATPES_USERNAME"),
            password=os.environ.get("MATPES_PASSWORD"),
            authSource="admin",
        )
        self.db = client.get_database("matpes")

    def get_json(self, functional: str, criteria: dict) -> list:
        """
        Args:
            functional (str): The name of the functional to query.
            criteria (dict): The criteria to query.
        """
        return list(self.db.get_collection(functional.lower()).find(criteria))

    def get_df(self, functional: str) -> pd.DataFrame:
        """
        Retrieve data for the given functional from the MongoDB database.

        Args:
            functional (str): The functional to query (e.g., 'PBE').

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        collection = self.db.get_collection(functional.lower())
        properties = [
            "matpes_id",
            "formula_pretty",
            "elements",
            "energy",
            "chemsys",
            "cohesive_energy_per_atom",
            "formation_energy_per_atom",
            "nsites",
            "nelements",
            "bandgap",
        ]
        return pd.DataFrame(
            collection.find(
                {},
                projection=properties,
            )
        )[properties]
