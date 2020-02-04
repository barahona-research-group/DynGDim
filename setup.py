#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (2, 7):
    sys.exit("Sorry, Python < 2.7 is not supported")

VERSION = imp.load_source("", "dyngdim/version.py").__version__

setup(
    name="dyngdim",
    author="Alexis Arnaudon and Robert Peach",
    author_email="alexis.arnaudon@epfl.ch, r.peach@imperial.ac.uk",
    version=VERSION,
    description="",
    install_requires=[
        'scipy',
        'numpy',
        'networkx',
        'matplotlib',
    ],
    packages=find_packages(),
)
