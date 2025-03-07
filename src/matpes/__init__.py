"""Tools for working with MatPES."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matpes")
except PackageNotFoundError:
    pass  # package not installed

MATPES_SRC = "https://s3.us-east-1.amazonaws.com/materialsproject-contribs/MatPES_2025_1"
