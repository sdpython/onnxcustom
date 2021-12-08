# -*- coding: utf-8 -*-
"""
Configuration for the documntation.
"""
import sys
import os
import warnings
import shutil
import pydata_sphinx_theme
from pyquickhelper.helpgen.default_conf import set_sphinx_variables

sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")


def callback_begin():
    source = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "examples", "images"))
    if not os.path.exists(source):
        raise FileNotFoundError("Folder %r not found." % source)
    dest = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "gyexamples", "images"))
    if not os.path.exists(dest):
        os.makedirs(dest)
    for img in os.listdir(source):
        ext = os.path.splitext(img)[-1]
        if ext not in {'.png', '.jpg'}:
            continue
        shutil.copy(os.path.join(source, img), dest)


set_sphinx_variables(__file__, "onnxcustom", "Xavier Dupré", 2021,
                     "pydata_sphinx_theme", pydata_sphinx_theme.get_html_theme_path(),
                     locals(), extlinks=dict(
                         issue=('https://github.com/sdpython/onnxcustom/issues/%s', 'issue')),
                     title="onnxcustom", book=True,
                     callback_begin=callback_begin)

extensions.extend([
    "sphinxcontrib.blockdiag",
    "myst_parser"
])

html_theme_options = {
    "github_user": "sdpython",
    "github_repo": "onnxcustom",
    "github_version": "master",
    "collapse_navigation": True,
    "show_nav_level": 2,
}

blog_root = "http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/"

html_css_files = ['my-styles.css']

html_logo = "phdoc_static/project_ico.png"
html_sidebars = {}
language = "en"

mathdef_link_only = True

custom_preamble = """\n
\\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
\\newcommand{\\N}[0]{\\mathbb{N}}
\\newcommand{\\indicatrice}[1]{\\mathbf{1\\!\\!1}_{\\acc{#1}}}
\\newcommand{\\infegal}[0]{\\leqslant}
\\newcommand{\\supegal}[0]{\\geqslant}
\\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
\\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
\\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
\\newcommand{\\loinormale}[2]{{\\cal N}\\pa{#1,#2}}
\\newcommand{\\independant}[0]{\\;\\makebox[3ex]
{\\makebox[0ex]{\\rule[-0.2ex]{3ex}{.1ex}}\\!\\!\\!\\!\\makebox[.5ex][l]
{\\rule[-.2ex]{.1ex}{2ex}}\\makebox[.5ex][l]{\\rule[-.2ex]{.1ex}{2ex}}} \\,\\,}
\\newcommand{\\esp}{\\mathbb{E}}
\\newcommand{\\pr}[1]{\\mathbb{P}\\pa{#1}}
\\newcommand{\\loi}[0]{{\\cal L}}
\\newcommand{\\vecteurno}[2]{#1,\\dots,#2}
\\newcommand{\\norm}[1]{\\left\\Vert#1\\right\\Vert}
\\newcommand{\\dans}[0]{\\rightarrow}
\\newcommand{\\partialfrac}[2]{\\frac{\\partial #1}{\\partial #2}}
\\newcommand{\\partialdfrac}[2]{\\dfrac{\\partial #1}{\\partial #2}}
\\newcommand{\\loimultinomiale}[1]{{\\cal M}\\pa{#1}}
\\newcommand{\\trace}[1]{tr\\pa{#1}}
\\newcommand{\\abs}[1]{\\left|#1\\right|}
"""

# \\usepackage{eepic}
imgmath_latex_preamble += custom_preamble
latex_elements['preamble'] += custom_preamble

intersphinx_mapping.update({
    'pandas_streaming':
        ('http://www.xavierdupre.fr/app/pyquickhelper/helpsphinx/', None),
})


epkg_dictionary.update({
    'C': 'https://en.wikipedia.org/wiki/C_(programming_language)',
    'C++': 'https://en.wikipedia.org/wiki/C%2B%2B',
    'chrome-tracing':
        'https://www.chromium.org/developers/how-tos/trace-event-profiling-tool',
    'cython': 'https://cython.org/',
    'docker': 'https://en.wikipedia.org/wiki/Docker_(software)',
    'DOT': 'https://www.graphviz.org/doc/info/lang.html',
    'ImageNet': 'http://www.image-net.org/',
    'LightGBM': 'https://lightgbm.readthedocs.io/en/latest/',
    'lightgbm': 'https://lightgbm.readthedocs.io/en/latest/',
    'mlprodict':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'netron': 'https://github.com/lutzroeder/netron',
    'NMF':
        'https://scikit-learn.org/stable/modules/generated/'
        'sklearn.decomposition.NMF.html',
    'numpy': 'https://numpy.org/',
    'nvprof': 'https://docs.nvidia.com/cuda/profiler-users-guide/index.html',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX': 'https://onnx.ai/',
    'ONNX operators':
        'https://github.com/onnx/onnx/blob/master/docs/Operators.md',
    'ONNX ML Operators':
        'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md',
    'ONNX Zoo': 'https://github.com/onnx/models',
    'onnxmltools': 'https://github.com/onnx/onnxmltools',
    'onnxruntime': 'https://microsoft.github.io/onnxruntime/',
    'onnxruntime-training':
        'https://github.com/microsoft/onnxruntime/tree/master/orttraining',
    'openmp': 'https://en.wikipedia.org/wiki/OpenMP',
    'pandas_streaming':
        'http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/index.html',
    'protobuf': 'https://developers.google.com/protocol-buffers',
    'py-spy': 'https://github.com/benfred/py-spy',
    'pyinstrument': 'https://github.com/joerick/pyinstrument',
    'python': 'https://www.python.org/',
    'pytorch': 'https://pytorch.org/',
    'scikit-learn': 'https://scikit-learn.org/stable/',
    'skorch': 'https://skorch.readthedocs.io/en/stable/',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
    'sphinx-gallery': 'https://github.com/sphinx-gallery/sphinx-gallery',
    'sqlite3': 'https://docs.python.org/3/library/sqlite3.html',
    'Stochastic Gradient Descent':
        'https://en.wikipedia.org/wiki/Stochastic_gradient_descent',
    'Tensor': 'https://en.wikipedia.org/wiki/Tensor',
    'tensor': 'https://en.wikipedia.org/wiki/Tensor',
    'torch': 'https://pytorch.org/',
    'tqdm': 'https://github.com/tqdm/tqdm',
    'xgboost': 'https://xgboost.readthedocs.io/en/latest/',
    'XGBoost': 'https://xgboost.readthedocs.io/en/latest/',
})

# APIs, links which should be replaced.

epkg_dictionary.update({
    'C_OrtValue':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/'
        'onnxmd/onnxruntime_python/ortvalue.html#c-class-ortvaluevector',
    'InferenceSession':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/inference.html'
        '#python-wrapper-for-inferencesession',
    'IOBinding':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#iobinding',
    'OnnxPipeline':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/'
        'sklapi/onnx_pipeline.html',
    'OrtModuleGraphBuilder':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html'
        "#ortmodulegraphbuilder",
    'OrtModuleGraphBuilderConfiguration':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html'
        "#ortmodulegraphbuilderconfiguration",
    'OrtDevice':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#ortdevice',
    'OrtValue':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#ortvalue',
    'OrtValueCache':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html#ortvaluecache',
    'OrtValueVector':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/training_session.html#ortvaluevector',
    'PartialGraphExecutionState':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html'
        "#partialgraphexecutionstate",
    'RunOptions':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/inference.html#runoptions',
    'SessionOptions':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/inference.html#sessionoptions',
    'SparseTensor':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#sparsetensor',
    'TrainingAgent':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html'
        "#trainingagent",
    'TreeEnsembleRegressor':
        'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md'
        '#ai.onnx.ml.TreeEnsembleRegressor',
})


nblinks = {
    'alter_pipeline_for_debugging':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/'
        'onnxcustom/helpers/pipeline.html'
        '#onnxcustom.helpers.pipeline.alter_pipeline_for_debugging',
}

warnings.filterwarnings("ignore", category=FutureWarning)
