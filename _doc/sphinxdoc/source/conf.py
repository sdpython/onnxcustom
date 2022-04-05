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


set_sphinx_variables(__file__, "onnxcustom", "Xavier Dupré", 2022,
                     "pydata_sphinx_theme", ['_static'],
                     locals(), extlinks=dict(
                         issue=('https://github.com/sdpython/onnxcustom/issues/%s', 'issue')),
                     title="onnxcustom", book=True,
                     callback_begin=callback_begin)

extensions.extend([
    "sphinxcontrib.blockdiag",
    "sphinx.ext.napoleon",
    "myst_parser",
    'mlprodict.npy.xop_sphinx',
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
onnx_doc_folder = os.path.join(os.path.dirname(__file__), 'api', 'onnxops')
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
    'torch': ('https://pytorch.org/docs/stable/', None),
    'mlprodict':
        ('http://www.xavierdupre.fr/app/mlprodict/helpsphinx/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'onnxcustom':
        ('http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pandas_streaming':
        ('http://www.xavierdupre.fr/app/pyquickhelper/helpsphinx/', None),
    'pyquickhelper':
        ('http://www.xavierdupre.fr/app/pyquickhelper/helpsphinx/', None),
    'python': (
        'https://docs.python.org/{.major}'.format(sys.version_info),
        None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'sklearn': ('https://scikit-learn.org/stable/', None)
})


epkg_dictionary.update({
    'C': 'https://en.wikipedia.org/wiki/C_(programming_language)',
    'C++': 'https://en.wikipedia.org/wiki/C%2B%2B',
    'chrome-tracing':
        'https://www.chromium.org/developers/how-tos/trace-event-profiling-tool',
    'cmake': 'https://cmake.org/',
    'COO': 'https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)',
    'CSR':
        'https://en.wikipedia.org/wiki/Sparse_matrix'
        '#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)',
    'cudnn': 'https://developer.nvidia.com/cudnn',
    'cython': 'https://cython.org/',
    'DLPack': 'https://github.com/dmlc/dlpack',
    'docker': 'https://en.wikipedia.org/wiki/Docker_(software)',
    'DOT': 'https://www.graphviz.org/doc/info/lang.html',
    'ImageNet': 'http://www.image-net.org/',
    'LightGBM': 'https://lightgbm.readthedocs.io/en/latest/',
    'lightgbm': 'https://lightgbm.readthedocs.io/en/latest/',
    'mlprodict':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'myst-parser': 'https://myst-parser.readthedocs.io/en/latest/',
    'nccl': 'https://developer.nvidia.com/nccl',
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
    'onnxruntime-extensions':
        'https://github.com/microsoft/onnxruntime-extensions',
    'onnxruntime-training':
        'https://github.com/microsoft/onnxruntime/tree/master/orttraining',
    'openmp': 'https://en.wikipedia.org/wiki/OpenMP',
    'openmpi': 'https://www.open-mpi.org/',
    'pandas_streaming':
        'http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/index.html',
    'protobuf': 'https://developers.google.com/protocol-buffers',
    'py-spy': 'https://github.com/benfred/py-spy',
    'pyinstrument': 'https://github.com/joerick/pyinstrument',
    'pyspark': 'https://spark.apache.org/docs/latest/api/python/',
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
    'tensorflow': 'https://www.tensorflow.org/',
    'tensorflow-onnx': 'https://github.com/onnx/tensorflow-onnx',
    'tf2onnx': 'https://github.com/onnx/tensorflow-onnx',
    'torch': 'https://pytorch.org/',
    'tqdm': 'https://github.com/tqdm/tqdm',
    'xgboost': 'https://xgboost.readthedocs.io/en/latest/',
    'XGBoost': 'https://xgboost.readthedocs.io/en/latest/',
    'WSL': 'https://docs.microsoft.com/en-us/windows/wsl/install',
    'zetane': 'https://github.com/zetane/viewer',
})

# APIs, links which should be replaced.

epkg_dictionary.update({
    'C_OrtDevice':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/'
        'onnxruntime_python/helpers.html#c-class-ortdevice',
    'C_OrtValue':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/'
        'onnxmd/onnxruntime_python/ortvalue.html#c-class-ortvaluevector',
    'C_SparseTensor':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#sparsetensor',
    'Contrib Operators':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_docs/ContribOperators.html',
    'Gemm':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-gemm-a-a-name-gemm-gemm-a',
    'If':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-if-a-a-name-if-if-a',
    'InferenceSession':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/inference.html'
        '#python-wrapper-for-inferencesession',
    'IOBinding':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#iobinding',
    'IR':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/IR.html',
    'Loop':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-loop-a-a-name-loop-loop-a',
    'OnnxPipeline':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/'
        'sklapi/onnx_pipeline.html',
    'OneHotEncoder':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators-ml.html?highlight=onehotencoding'
        '#a-name-ai-onnx-ml-onehotencoder-a-a-name-ai-onnx-'
        'ml-onehotencoder-ai-onnx-ml-onehotencoder-a',
    'ORTModule':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/'
        'api/onnxruntime_python/training_torch.html#ortmodule',
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
    'Scan':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-scan-a-a-name-scan-scan-a',
    'SessionIOBinding':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/'
        'onnxruntime_python/inference.html#'
        'onnxruntime.capi._pybind_state.SessionIOBinding',
    'SessionOptions':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/inference.html#sessionoptions',
    'SparseTensor':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#sparsetensor',
    'TfIdfVectorizer':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#'
        'a-name-tfidfvectorizer-a-a-name-tfidfvectorizer-tfidfvectorizer-a',
    'TopK':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-topk-a-a-name-topk-topk-a',
    'TrainingAgent':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html'
        "#trainingagent",
    'TrainingParameters':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/'
        'onnxruntime_python/training.html#trainingparameters',
    'TrainingSession':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/'
        'onnxruntime_python/training.html#onnxruntime.TrainingSession',
    'Transpose':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html'
        '#a-name-transpose-a-a-name-transpose-transpose-a',
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
