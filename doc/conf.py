# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, basedir)
sys.path.insert(0, os.path.join(basedir, "pbnn"))
print(sys.path)

project = "pbnn"
copyright = "Safran Group"
author = "Safran Group"

release = "0.0.1"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.duration',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'myst_nb',
    # 'myst_parser', # imported by myst_nb
    # 'sphinxcontrib.apidoc', # autoapi is better
    'sphinx.ext.autosummary',
    # 'sphinxcontrib.bibtex'
]

# bibtex_bibfiles = ['refs.bib']
# bibtex_encoding = 'latin'
# bibtex_default_style = 'unsrt'

# -----------------------------------------------------------------------------#
# sphinx.ext.intersphinx options
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'pytest': ('https://pytest.org/en/stable/', None),
#     'numpy': ('https://numpy.org/doc/stable/', None),
# }
# sphinx.ext.extlinks options
# extlinks_detect_hardcoded_links = True
# sphinx.ext.graphviz options
# graphviz_output_format = 'svg'
# myst_parser options
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
    '.md': 'myst-nb',
}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 7

numfig = True
autosummary_generate = True

templates_path = ["_templates"]
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_css_files = ["custom.css"]
html_static_path = ["_static"]
html_title = "pbnn"
html_css_files = ["custom.css"]
html_context = {"default_mode": "light"}

autodoc_mock_imports = ["jax"]
autodoc_member_order = "bysource"
