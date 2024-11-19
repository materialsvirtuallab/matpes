"""Package configuration."""

from __future__ import annotations

from setuptools import setup

with open("README.md") as file:
    long_description = file.read()

setup(
    name="python_template",
    version="0.0.1",
    python_requires=">=3.8",
    install_requires=[],
    extras_require={},
    package_data={},
    author="Author list",
    author_email="ongsp@ucsd.edu",
    maintainer="Shyue Ping Ong",
    maintainer_email="ongsp@ucsd.edu",
    url="https://materialsvirtuallab.org",
    description="Template",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
