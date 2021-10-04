#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))
sys.path.append(
    os.path.abspath(os.path.dirname(__file__)))
from onnxcustom import __version__  # noqa
from github_link import make_linkcode_resolve  # noqa

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_gallery.gen_gallery',
    'alabaster',
    'matplotlib.sphinxext.plot_directive',
    'pyquickhelper.sphinxext.sphinx_cmdref_extension',
    'pyquickhelper.sphinxext.sphinx_collapse_extension',
    'pyquickhelper.sphinxext.sphinx_docassert_extension',
    'pyquickhelper.sphinxext.sphinx_epkg_extension',
    'pyquickhelper.sphinxext.sphinx_exref_extension',
    'pyquickhelper.sphinxext.sphinx_faqref_extension',
    'pyquickhelper.sphinxext.sphinx_gdot_extension',
    'pyquickhelper.sphinxext.sphinx_runpython_extension',
    'sphinxcontrib.blockdiag',
]

templates_path = ['_templates']
html_logo = '_static/logo.png'
source_suffix = '.rst'
master_doc = 'index'
project = 'onnxcustom'
copyright = '2020, Xavier Dupré, ...'
author = 'Xavier Dupré'
version = __version__
release = __version__
language = 'en'
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = True

import alabaster  # noqa
html_theme = "alabaster"
html_theme_path = [alabaster.get_path()]

html_theme_options = {}
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

htmlhelp_basename = 'onnxcustom'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc,
     'onnxcustom.tex',
     'Documentation',
     'Xavier Dupré',
     'manual'),
]

latex_engine = 'xelatex'
latex_elements = {
    'preamble': """\n
\\newcommand{\\norm}[1]{\\left\\Vert#1\\right\\Vert}
"""
}

texinfo_documents = [
    (master_doc, 'onnxcustom', 'onnxcustom Documentation',
     author, 'onnxcustom', 'One line description of project.',
     'Miscellaneous'),
]

linkcode_resolve = make_linkcode_resolve(
    'onnxcustom',
    'https://github.com/sdpython/onnxcustom/blob/{revision}/'
    '{package}/{path}#L{lineno}')

intersphinx_mapping = {
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
    'mlinsights': (
        'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/', None),
    'mlprodict': (
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pyquickhelper': (
        'http://www.xavierdupre.fr/app/pyquickhelper/helpsphinx/', None),
    'onnxmltools': (
        'http://www.xavierdupre.fr/app/onnxmltools/helpsphinx/',
        None),
    'onnxruntime': (
        'http://www.xavierdupre.fr/app/onnxruntime/helpsphinx/',
        None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'scikit-learn': (
        'https://scikit-learn.org/stable/',
        None),
    'sklearn': (
        'https://scikit-learn.org/stable/',
        None),
    'skl2onnx': (
        'http://www.xavierdupre.fr/app/sklearn-onnx/helpsphinx/',
        None),
    'sklearn-onnx': (
        'http://www.xavierdupre.fr/app/sklearn-onnx/helpsphinx/',
        None),
}

sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': os.path.join(os.path.dirname(__file__), '../examples'),
    # path where to save gallery generated examples
    'gallery_dirs': 'auto_examples',
    'capture_repr': ('_repr_html_', '__repr__'),
    'ignore_repr_types': r'matplotlib.text|matplotlib.axes',
    'binder': {
        'org': 'sdpython',
        'repo': 'onnxcustom',
        'binderhub_url': 'https://mybinder.org',
        'branch': 'master',
        'dependencies': os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'requirements.txt')),
        'use_jupyter_lab': True
    },
}

epkg_dictionary = {
    'C': 'https://en.wikipedia.org/wiki/C_(programming_language)',
    'C++': 'https://en.wikipedia.org/wiki/C%2B%2B',
    'cython': 'https://cython.org/',
    'DOT': 'https://www.graphviz.org/doc/info/lang.html',
    'ImageNet': 'http://www.image-net.org/',
    'LightGBM': 'https://lightgbm.readthedocs.io/en/latest/',
    'lightgbm': 'https://lightgbm.readthedocs.io/en/latest/',
    'mlprodict':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'NMF':
        'https://scikit-learn.org/stable/modules/generated/'
        'sklearn.decomposition.NMF.html',
    'numpy': 'https://numpy.org/',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX': 'https://onnx.ai/',
    'ONNX operators':
        'https://github.com/onnx/onnx/blob/master/docs/Operators.md',
    'ONNX ML Operators':
        'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md',
    'ONNX Zoo': 'https://github.com/onnx/models',
    'onnxmltools': 'https://github.com/onnx/onnxmltools',
    'OnnxPipeline':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/'
        'sklapi/onnx_pipeline.html?highlight=onnxpipeline',
    'onnxruntime': 'https://microsoft.github.io/onnxruntime/',
    'onnxruntime-training':
        'https://github.com/microsoft/onnxruntime/tree/master/orttraining',
    'openmp': 'https://en.wikipedia.org/wiki/OpenMP',
    'pyinstrument': 'https://github.com/joerick/pyinstrument',
    'python': 'https://www.python.org/',
    'pytorch': 'https://pytorch.org/',
    'scikit-learn': 'https://scikit-learn.org/stable/',
    'skorch': 'https://skorch.readthedocs.io/en/stable/',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
    'sphinx-gallery': 'https://github.com/sphinx-gallery/sphinx-gallery',
    'Stochastic Gradient Descent':
        'https://en.wikipedia.org/wiki/Stochastic_gradient_descent',
    'TreeEnsembleRegressor':
        'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md'
        '#ai.onnx.ml.TreeEnsembleRegressor',
    'xgboost': 'https://xgboost.readthedocs.io/en/latest/',
    'XGBoost': 'https://xgboost.readthedocs.io/en/latest/',
}

warnings.filterwarnings("ignore", category=FutureWarning)
