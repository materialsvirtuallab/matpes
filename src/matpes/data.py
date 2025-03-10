"""Methods for working with MatPES data downloads."""

from __future__ import annotations

import gzip
import json
import os
from typing import Literal

import requests
from tqdm import tqdm

from matpes import MATPES_SRC


def get_data(
    functional: Literal["PBE", "R2SCAN"] = "PBE",
    version: str = "2025.1",
    return_entries: bool = True,
    download_atoms: bool = False,
):
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
        return_entries (bool): Whether to return the deserialized entries from JSON or not. Defaults to True.
        include_atoms (bool): Whether to download the atomic reference file.

    Returns:
        dict: A dictionary representation of the JSON dataset contents.

    Raises:
        RuntimeError: If the file download fails or the remote source is
                      inaccessible.
    """
    fnames = [f"MatPES-{functional.upper()}-{version}.json.gz"]
    if download_atoms:
        fnames.append(f"MatPES-{functional.upper()}-atoms.json.gz")

    for fname in fnames:
        if not os.path.exists(fname):
            url = f"{MATPES_SRC}/{fname}"
            print(f"Downloading from {url}...")

            # Streaming, so we can iterate over the response.
            response = requests.get(url, stream=True)

            # Sizes in bytes.
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar, open(fname, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            if total_size != 0 and progress_bar.n != total_size:
                raise RuntimeError(f"Failed to download {url}. Status code: {response.status_code}")
    if return_entries:
        with gzip.open(fnames[0], "r") as f:
            return json.load(f)
    return None
