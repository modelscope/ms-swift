# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# import sphinx_book_theme

sys.path.insert(0, os.path.abspath('../../'))
# -- Project information -----------------------------------------------------

project = 'swift'
copyright = '2022-2024, Alibaba ModelScope'
author = 'ModelScope Authors'
version_file = '../../swift/version.py'
html_theme = 'sphinx_rtd_theme'
language = 'zh_CN'


def get_version():
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


# The full version, including alpha/beta/rc tags
version = get_version()
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
    'sphinx_copybutton',
    'myst_parser',
]

# build the templated autosummary files
autosummary_generate = True
numpydoc_show_class_members = False

# Enable overriding of function signatures in the first line of the docstring.
autodoc_docstring_signature = True

# Disable docstring inheritance
autodoc_inherit_docstrings = False

# Show type hints in the description
autodoc_typehints = 'description'

# Add parameter types if the parameter is documented in the docstring
autodoc_typehints_description_target = 'documented_params'

autodoc_default_options = {
    'member-order': 'bysource',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# The master toctree document.
root_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'source/.ipynb_checkpoints', 'source/api/generated', 'Thumbs.db', '.DS_Store']
# A list of glob-style patterns [1] that are used to find source files.
# They are matched against the source file names relative to the source directory,
# using slashes as directory separators on all platforms.
# The default is **, meaning that all files are recursively included from the source directory.
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_book_theme'
# html_theme_path = [sphinx_book_theme.get_html_theme_path()]
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_css_files = ['css/readthedocs.css']

# -- Options for HTMLHelp output ---------------------------------------------
# Output file base name for HTML help builder.

# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

myst_enable_extensions = [
    'amsmath',
    'dollarmath',
    'colon_fence',
]
