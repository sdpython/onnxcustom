"""
.. _example-lightgbm-reg-one-off:

Batch predictions vs one-off predictions
========================================

.. index:: LightGBM

The goal is to compare the processing time between batch predictions
and one-off prediction for the same number of predictions
on trees. onnxruntime parallelizes the prediction by trees
or by rows. The rule is fixed and cannot be changed but it seems
to have some loopholes.

.. contents::
    :local:

Train a LGBMRegressor
+++++++++++++++++++++
"""
import warnings
import time
import os
from packaging.version import Version
import numpy
from pandas import DataFrame
import onnx
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightgbm import LGBMRegressor
from onnxruntime import InferenceSession
from skl2onnx import update_registered_converter, to_onnx
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes  # noqa
from onnxmltools import __version__ as oml_version
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm  # noqa


N = 1000
X = numpy.random.randn(N, 1000)
y = (numpy.random.randn(N) +
     numpy.random.randn(N) * 100 * numpy.random.randint(0, 1, N))

filenames = [f"plot_lightgbm_regressor_1000_{X.shape[1]}.onnx",
             f"plot_lightgbm_regressor_10_{X.shape[1]}.onnx",
             f"plot_lightgbm_regressor_2_{X.shape[1]}.onnx"]

if not os.path.exists(filenames[0]):
    print(f"training with shape={X.shape}")
    reg_1000 = LGBMRegressor(n_estimators=1000)
    reg_1000.fit(X, y)
    reg_10 = LGBMRegressor(n_estimators=10)
    reg_10.fit(X, y)
    reg_2 = LGBMRegressor(n_estimators=2)
    reg_2.fit(X, y)
    print("done.")
else:
    print("A model was already trained. Reusing it.")

######################################
# Register the converter for LGBMRegressor
# ++++++++++++++++++++++++++++++++++++++++


def skl2onnx_convert_lightgbm(scope, operator, container):
    options = scope.get_options(operator.raw_operator)
    if 'split' in options:
        if Version(oml_version) < Version('1.9.2'):
            warnings.warn(
                "Option split was released in version 1.9.2 but %s is "
                "installed. It will be ignored." % oml_version)
        operator.split = options['split']
    else:
        operator.split = None
    convert_lightgbm(scope, operator, container)


update_registered_converter(
    LGBMRegressor, 'LightGbmLGBMRegressor',
    calculate_linear_regressor_output_shapes,
    skl2onnx_convert_lightgbm,
    options={'split': None})

##################################
# Convert
# +++++++
#
# We convert the same model following the two scenarios, one single
# TreeEnsembleRegressor node, or more. *split* parameter is the number of
# trees per node TreeEnsembleRegressor.

if not os.path.exists(filenames[0]):
    model_onnx_1000 = to_onnx(reg_1000, X[:1].astype(numpy.float32),
                              target_opset={'': 17, 'ai.onnx.ml': 3})
    with open(filenames[0], "wb") as f:
        f.write(model_onnx_1000.SerializeToString())
    model_onnx_10 = to_onnx(reg_10, X[:1].astype(numpy.float32),
                              target_opset={'': 17, 'ai.onnx.ml': 3})
    with open(filenames[1], "wb") as f:
        f.write(model_onnx_10.SerializeToString())
    model_onnx_2 = to_onnx(reg_2, X[:1].astype(numpy.float32),
                              target_opset={'': 17, 'ai.onnx.ml': 3})
    with open(filenames[2], "wb") as f:
        f.write(model_onnx_10.SerializeToString())
else:
    with open(filenames[0], "rb") as f:
        model_onnx_1000 = onnx.load(f)
    with open(filenames[1], "rb") as f:
        model_onnx_10 = onnx.load(f)
    with open(filenames[2], "rb") as f:
        model_onnx_2 = onnx.load(f)

sess_1000 = InferenceSession(model_onnx_1000.SerializeToString(),
                             providers=['CPUExecutionProvider'])
sess_10 = InferenceSession(model_onnx_10.SerializeToString(),
                           providers=['CPUExecutionProvider'])
sess_2 = InferenceSession(model_onnx_2.SerializeToString(),
                           providers=['CPUExecutionProvider'])

##########################
# Processing time
# +++++++++++++++
#

repeat = 5
data = []
for N in tqdm(list(range(10, 100, 10)) +
              list(range(100, 1000, 100)) + 
              list(range(1000, 10001, 1000))):

    X32 = numpy.random.randn(N, X.shape[1]).astype(numpy.float32)
    obs = dict(N=N)
    for sess, T in [(sess_1000, 1000), (sess_10, 10), (sess_2, 2)]:
        times = []
        for _ in range(repeat):
            begin = time.perf_counter()
            sess.run(None, {'X': X32})
            end = time.perf_counter() - begin
            times.append(end / X32.shape[0])
        times.sort()
        obs[f"batch-{T}"] = sum(times[1:-1]) / (len(times) - 2)

        times = []
        for _ in range(repeat):
            begin = time.perf_counter()
            for i in range(X32.shape[0]):
                sess.run(None, {'X': X32[i: i+1]})
            end = time.perf_counter() - begin
            times.append(end / X32.shape[0])
        times.sort()
        obs[f"one-off-{T}"] = sum(times[1:-1]) / (len(times) - 2)
    data.append(obs)

df = DataFrame(data).set_index("N")
print(df)

########################################
# Plots.
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

df[["batch-1000", "one-off-1000"]].plot(
    ax=ax[0], title="Processing time per observation\n1000 Trees",
    logy=True, logx=True)
df[["batch-10", "one-off-10"]].plot(
    ax=ax[1], title="Processing time per observation\n10 Trees",
    logy=True, logx=True)
df[["batch-2", "one-off-2"]].plot(
    ax=ax[2], title="Processing time per observation\n2 Trees",
    logy=True, logx=True)

##########################################
# Conclusion
#
# The first graph shows a huge drop the prediction time by batch.
# It means the parallelization is triggered. It may have been triggered
# sooner on this machine but this decision could be different on another one.
# An approach like the one TVM chose could be a good answer. If the model
# must be fast, then it is worth benchmarking many strategies to parallelize
# until the best one is found on a specific machine.

# plt.show()
