import pathlib
import sys
import os
# import flatiron_tk
sys.path.insert(0, os.path.abspath("../"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'flatiron_tk'
copyright = '2024, FLOWLab (PI: Debanjan Mukherjee), University of Colorado Boulder'
author = 'FLOWLab Team'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',
              'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []

mathjax3_config = {'chtml': {'displayAlign': 'center'}}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
