#Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os, sys
sys.path.insert(0, os.path.abspath('../../charmm'))

print(sys.path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'charmm'
copyright = '2022, Samarjeet Prasad'
author = 'Samarjeet Prasad'
release = '0.0.1'




# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 
               'sphinx.ext.duration',
               'sphinx.ext.autodoc',
               'sphinx.ext.autosummary', 
               'sphinx.ext.todo', 
               ]

autosummary_generate=True

templates_path = ['_templates']
exclude_patterns = []

todo_include_todos = True # Include todo blocks found in the rst files in the
                          # compiled documentation. Turn off for a release, 
                          # cleaner doc.

add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_sidebars= {
      '*' : [
         'localtoc.html', 'searchbox.html' ], 
      '_autosummary/*': [
         'globaltoc.html', 'localtoc.html', 'searchbox.html' ],
      }
