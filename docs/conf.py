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
from sphinx_gallery.sorting import FileNameSortKey
import pyvista
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'megicparc'
copyright = '2023, Sara Sommariva'
author = 'Sara Sommariva'

# The full version, including alpha/beta/rc tags
release = '1.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    	      'sphinx.ext.autosummary', 
	      'sphinx_gallery.gen_gallery']
autosummary_generate = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'

html_theme_options = {
    'navbar_sidebarrel': False,
    'navbar_pagenav': False,
    'globaltoc_includehidden': False,
    'navbar_links': [
        ("API", "api"),
	("Examples", "auto_examples/index"),
	("Reproducing paper results", "auto_paper/index"),
#        ("GitHub", "https://github.com/elisabettavallarino/transfreq", True)
    ],
    'bootswatch_theme': "united"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

examples_dirs = ['../examples', '../code_paper']
gallery_dirs = ['auto_examples', 'auto_paper', ]

pyvista.OFF_SCREEN = False
pyvista.BUILDING_GALLERY = True

sphinx_gallery_conf = {
     'examples_dirs': examples_dirs,   # path to your example scripts
     'gallery_dirs': gallery_dirs,  # path to where to save gallery generated output
     'image_scrapers': ('matplotlib', 'pyvista'), # figures extension to be embedded
     'default_thumb_file': os.path.join('_static', 'ex_parc.png'),
     'within_subsection_order': FileNameSortKey
}

#CONFIG['MPL_RENDER'] = True # Embed visbrain figures in the documentation
