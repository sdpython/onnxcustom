"""

.. _l-orttraining-benchmark:

Benchmark, comparison scikit-learn - onnxruntime-training
=========================================================

The benchmark compares the processing time between :epkg:`scikit-learn`
and :epkg:`onnxruntime-training` on a linear regression and a neural network.
It uses the model trained in :ref:`l-orttraining-nn-gpu`.


.. contents::
    :local:

First comparison: neural network
++++++++++++++++++++++++++++++++

"""
import warnings
from pprint import pprint
import time
import numpy
from pandas import DataFrame
from onnxruntime import get_device
from pyquickhelper.pycode.profiling import profile, profile2graph
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from mlprodict.onnx_conv import to_onnx
from onnxcustom.utils.onnx_orttraining import (
    add_loss_output, get_train_initializer)
from onnxcustom.training.optimizers import OrtGradientOptimizer


X, y = make_regression(1000, n_features=100, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

########################################
# Common parameters and model

batch_size = 15
max_iter = 100

nn = MLPRegressor(hidden_layer_sizes=(50, 10), max_iter=max_iter,
                  solver='sgd', learning_rate_init=1e-4,
                  n_iter_no_change=max_iter * 3, batch_size=batch_size)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nn.fit(X_train, y_train)

########################################
# Conversion to ONNX and trainer initialization

onx = to_onnx(nn, X_train[:1].astype(numpy.float32), target_opset=15)
onx_train = add_loss_output(onx)

weights = get_train_initializer(onx)
pprint(list((k, v[0].shape) for k, v in weights.items()))

train_session = OrtGradientOptimizer(
    onx_train, list(weights), device='cpu', learning_rate=1e-4,
    warm_start=False, max_iter=max_iter, batch_size=batch_size)


def benchmark(skl_model, train_session, name, verbose=True):

    print("[benchmark] %s" % name)
    begin = time.perf_counter()
    skl_model.fit(X, y)
    duration_skl = time.perf_counter() - begin
    length_skl = len(skl_model.loss_curve_)
    print("[benchmark] skl=%r iterations - %r seconds" % (
        length_skl, duration_skl))

    begin = time.perf_counter()
    train_session.fit(X, y)
    duration_ort = time.perf_counter() - begin
    length_ort = len(train_session.train_losses_)
    print("[benchmark] ort=%r iterations - %r seconds" % (
        length_ort, duration_ort))

    return dict(skl=duration_skl, ort=duration_ort, name=name,
                iter_skl=length_skl, iter_ort=length_ort)


benches = [benchmark(nn, train_session, name='NN-CPU')]

######################################
# Profiling
# +++++++++


def clean_name(text):
    pos = text.find('onnxruntime')
    if pos >= 0:
        return text[pos:]
    pos = text.find('onnxcustom')
    if pos >= 0:
        return text[pos:]
    pos = text.find('site-packages')
    if pos >= 0:
        return text[pos:]
    return text


ps = profile(lambda: benchmark(nn, train_session, name='NN-CPU'))[0]
root, nodes = profile2graph(ps, clean_text=clean_name)
text = root.to_text()
print(text)

######################################
# if GPU is available
# +++++++++++++++++++

if get_device() == 'GPU':

    train_session = OrtGradientOptimizer(
        onx_train, list(weights), device='cuda', learning_rate=1e-4,
        warm_start=False, max_iter=200, batch_size=batch_size)

    benches.append(benchmark(nn, train_session, name='NN-GPU'))

######################################
# Linear Regression
# +++++++++++++++++

lr = MLPRegressor(hidden_layer_sizes=tuple(), max_iter=max_iter,
                  solver='sgd', learning_rate_init=1e-4,
                  n_iter_no_change=max_iter * 3, batch_size=batch_size)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lr.fit(X, y)


onx = to_onnx(nn, X_train[:1].astype(numpy.float32), target_opset=15)
onx_train = add_loss_output(onx)

inits = get_train_initializer(onx)
weights = {k: v for k, v in inits.items() if k != "shape_tensor"}
pprint(list((k, v[0].shape) for k, v in weights.items()))

train_session = OrtGradientOptimizer(
    onx_train, list(weights), device='cpu', learning_rate=1e-4,
    warm_start=False, max_iter=max_iter, batch_size=batch_size)

benches.append(benchmark(lr, train_session, name='LR-CPU'))

if get_device() == 'GPU':

    train_session = OrtGradientOptimizer(
        onx_train, list(weights), device='cuda', learning_rate=5e-4,
        warm_start=False, max_iter=200, batch_size=batch_size)

    benches.append(benchmark(nn, train_session, name='LR-GPU'))


######################################
# GPU profiling
# +++++++++++++

if get_device() == 'GPU':
    ps = profile(lambda: benchmark(nn, train_session, name='LR-GPU'))[0]
    root, nodes = profile2graph(ps, clean_text=clean_name)
    text = root.to_text()
    print(text)

######################################
# Graphs
# ++++++
#
# Dataframe first.

df = DataFrame(benches).set_index('name')
df

#######################################
# text output

print(df)

#######################################
# Graphs.

ax = df[['skl', 'ort']].plot.bar(title="Processing time")
ax.tick_params(axis='x', rotation=30)

# import matplotlib.pyplot as plt
# plt.show()
