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
Ntrees = [10, 100, 200]
X = numpy.random.randn(N, 1000)
y = (numpy.random.randn(N) +
     numpy.random.randn(N) * 100 * numpy.random.randint(0, 1, N))

filenames = [f"plot_lightgbm_regressor_{nt}_{X.shape[1]}.onnx"
             for nt in Ntrees]

regs = []
for nt, filename in zip(Ntrees, filenames):
    if not os.path.exists(filename):
        print(f"training with shape={X.shape} and {nt} trees")
        r = LGBMRegressor(n_estimators=nt).fit(X, y)
        regs.append(r)
        print("done.")
    else:
        regs.append(None)

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

models_onnx = []
for i, filename in enumerate(filenames):
    print(i, filename)
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            model_onnx = onnx.load(f)
        models_onnx.append(model_onnx)
    else:
        model_onnx = to_onnx(regs[i], X[:1].astype(numpy.float32),
                             target_opset={'': 17, 'ai.onnx.ml': 3})
        models_onnx.append(model_onnx)
        with open(filename, "wb") as f:
            f.write(model_onnx.SerializeToString())

sesss = [InferenceSession(m.SerializeToString(),
                          providers=['CPUExecutionProvider'])
         for m in models_onnx]

##########################
# Processing time
# +++++++++++++++
#

repeat = 7
data = []
for N in tqdm(list(range(10, 100, 10)) +
              list(range(100, 1000, 100)) +
              list(range(1000, 10001, 1000))):

    X32 = numpy.random.randn(N, X.shape[1]).astype(numpy.float32)
    obs = dict(N=N)
    for sess, T in zip(sesss, Ntrees):
        times = []
        for _ in range(repeat):
            begin = time.perf_counter()
            sess.run(None, {'X': X32})
            end = time.perf_counter() - begin
            times.append(end / X32.shape[0])
        times.sort()
        obs[f"batch-{T}"] = sum(times[2:-2]) / (len(times) - 4)

        times = []
        for _ in range(repeat):
            begin = time.perf_counter()
            for i in range(X32.shape[0]):
                sess.run(None, {'X': X32[i: i + 1]})
            end = time.perf_counter() - begin
            times.append(end / X32.shape[0])
        times.sort()
        obs[f"one-off-{T}"] = sum(times[2:-2]) / (len(times) - 4)
    data.append(obs)

df = DataFrame(data).set_index("N")
df.reset_index(drop=False).to_csv(
    "plot_gexternal_lightgbm_reg_per.csv", index=False)
print(df)

########################################
# Plots.
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

for i, T in enumerate(Ntrees):
    df[[f"batch-{T}", f"one-off-{T}"]].plot(
        ax=ax[i], title=f"Processing time per observation\n{T} Trees",
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
