"""Methods for working with MatPES data downloads."""

from __future__ import annotations

import gzip
import json
import os
from typing import Literal

import requests

from matpes import MATPES_SRC


def get_data(functional: Literal["PBE", "R2SCAN"] = "PBE", version="20240214"):
    """
    Downloads and reads a JSON dataset file if not already present locally. The file
    is expected to be hosted at a remote location, and the function will use the
    specified functional and version to construct the file name. If the file is
    not found locally, it will attempt to download the file, save it locally in
    compressed format, and then load its contents.

    Parameters:
        functional (str): The functional type used for labeling the dataset.
                          Defaults to "PBE".
        version (str): The version string for the dataset. Defaults to "20240214".

    Returns:
        dict: A dictionary representation of the JSON dataset contents.

    Raises:
        RuntimeError: If the file download fails or the remote source is
                      inaccessible.
    """
    fname = f"MatPES-{functional.upper()}-{version}.json.gz"

    if not os.path.exists(fname):
        url = f"{MATPES_SRC}/{fname}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(fname, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        else:
            raise RuntimeError(f"Failed to download {url}. Status code: {response.status_code}")
    with gzip.open(fname, "r") as f:
        return json.load(f)
