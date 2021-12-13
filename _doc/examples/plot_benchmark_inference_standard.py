"""
.. _benchmark-inference-sklearn:

Benchmark inference for scikit-learn models
===========================================

This short code compares the execution of a couple of runtime
for inference including :epkg:`onnxruntime`. It uses examples
`Measure ONNX runtime performances
<http://www.xavierdupre.fr/app/mlprodict/helpsphinx/
gyexamples/plot_onnx_benchmark.html>`_. It is an automated process
to compare the performance of a model against :epkg:`scikit-learn`.
This model is a simple model taken from all implemented by
:epkg:`scikit-learn`.

.. contents::
    :local:

Linear Regression
+++++++++++++++++

"""
from pandas import read_csv
from mlprodict.cli import validate_runtime
from mlprodict.plotting.plotting import plot_validate_benchmark

res = validate_runtime(
    verbose=1,
    out_raw="data.csv", out_summary="summary.csv",
    benchmark=True, dump_folder="dump_errors",
    runtime=['python', 'onnxruntime1'],
    models=['LinearRegression'],
    skip_models=['LinearRegression[m-reg]'],
    n_features=[10, 50], dtype="32",
    out_graph="bench.png",
    opset_min=15, opset_max=15,
    time_kwargs={
        1: {"number": 50, "repeat": 50},
        10: {"number": 25, "repeat": 25},
        100: {"number": 20, "repeat": 20},
        1000: {"number": 20, "repeat": 20},
        10000: {"number": 10, "repeat": 10},
    }
)

results = read_csv('summary.csv')
results

###########################################
# Graph.

_, ax = plot_validate_benchmark(results)
ax

# import matplotlib.pyplot as plt
# plt.show()
