#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Imports
#

import sys
import os

from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__))))



# -- RTD configuration ------------------------------------------------

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# This is used for linking and such so we link to the thing we're building
rtd_version = os.environ.get('READTHEDOCS_VERSION', 'latest')
if rtd_version not in ['stable', 'latest']:
    rtd_version = 'stable'

# -- General configuration ------------------------------------------------


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'MLBench Benchmarks'
copyright = '2018 MLBench development team'



autoclass_content = 'both'

intersphinx_mapping = {}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output ----------------------------------------------

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
else:
    html_theme = 'default'

# Output file base name for HTML help builder.
htmlhelp_basename = 'MLBench_Benchmarks'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', 'MLBench_DaMLBench_Benchmarksshboard.tex', 'MLBench Benchmarks Documentation',
   'MLBench development team', 'manual'),
]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'MLBench_Benchmarks', 'MLBench Benchmarks Documentation',
   'MLBench Benchmarks development team', 'MLBench_Benchmarks', 'One line description of project.',
   'Miscellaneous'),
]

# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = 'MLBench Benchmarks'
epub_author = 'MLBench development team'
epub_publisher = 'MLBench development team'
epub_copyright = '2018, MLBench development team'

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Custom Document processing ----------------------------------------------

import gensidebar
gensidebar.generate_sidebar(globals(), 'mlbench_benchmarks')

import sphinx.addnodes
import docutils.nodes

def process_child(node):
    '''This function changes class references to not have the
       intermediate module name by hacking at the doctree'''

    # Edit descriptions to be nicer
    if isinstance(node, sphinx.addnodes.desc_addname):
        if len(node.children) == 1:
            child = node.children[0]
            text = child.astext()

    # Edit literals to be nicer
    elif isinstance(node, docutils.nodes.literal):
        child = node.children[0]
        text = child.astext()

    for child in node.children:
        process_child(child)

def doctree_read(app, doctree):
    for child in doctree.children:
        process_child(child)

def setup(app):
    app.connect('doctree-read', doctree_read)
