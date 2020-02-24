#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="dyngdim",
    author="Alexis Arnaudon and Robert Peach",
    author_email="alexis.arnaudon@epfl.ch, r.peach@imperial.ac.uk",
    version="0.0.1",
    description="Code to compute relative/local and global dimensions from diffusion dynamics on graphs",
    install_requires=["scipy", "numpy", "networkx", "matplotlib", "tqdm"],
    packages=find_packages(),
)
