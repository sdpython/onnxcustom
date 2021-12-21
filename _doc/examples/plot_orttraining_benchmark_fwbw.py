"""

.. _l-orttraining-benchmark-fwbw:

Benchmark, comparison scikit-learn - forward-backward
=====================================================

The benchmark compares the processing time between :epkg:`scikit-learn`
and :epkg:`onnxruntime-training` on a linear regression and a neural network.
It replicates the benchmark implemented in :ref:`l-orttraining-benchmark`
but uses the forward backward approach developped in
:ref:`l-orttraining-linreg-fwbw`.

.. contents::
    :local:

First comparison: neural network
++++++++++++++++++++++++++++++++

"""
import warnings
import time
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame
from onnxruntime import get_device
from pyquickhelper.pycode.profiling import profile, profile2graph
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from mlprodict.onnx_conv import to_onnx
from onnxcustom.utils.onnx_helper import onnx_rename_weights
from onnxcustom.training.optimizers_partial import (
    OrtGradientForwardBackwardOptimizer)


X, y = make_regression(1000, n_features=100, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

########################################
# Benchmark function.


def benchmark(X, y, skl_model, train_session, name, verbose=True):
    """
    :param skl_model: model from scikit-learn
    :param train_session: instance of OrtGradientForwardBackwardOptimizer
    :param name: experiment name
    :param verbose: to debug
    """
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
    print("[benchmark] ort=%r iteration - %r seconds" % (
        length_ort, duration_ort))

    return dict(skl=duration_skl, ort=duration_ort, name=name,
                iter_skl=length_skl, iter_ort=length_ort,
                losses_skl=skl_model.loss_curve_,
                losses_ort=train_session.train_losses_)


########################################
# Common parameters and model

batch_size = 15
max_iter = 100

nn = MLPRegressor(hidden_layer_sizes=(50, 10), max_iter=max_iter,
                  solver='sgd', learning_rate_init=5e-4, alpha=0,
                  n_iter_no_change=max_iter * 3, batch_size=batch_size,
                  nesterovs_momentum=False, momentum=0,
                  learning_rate="invscaling")

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nn.fit(X_train, y_train)

########################################
# Conversion to ONNX and trainer initialization

onx = to_onnx(nn, X_train[:1].astype(numpy.float32), target_opset=15)
onx = onnx_rename_weights(onx)

train_session = OrtGradientForwardBackwardOptimizer(
    onx, device='cpu', learning_rate=1e-5,
    warm_start=False, max_iter=max_iter, batch_size=batch_size)


benches = [benchmark(X_train, y_train, nn, train_session, name='NN-CPU')]

######################################
# Profiling
# +++++++++


def clean_name(text):
    pos = text.find('onnxruntime')
    if pos >= 0:
        return text[pos:]
    pos = text.find('sklearn')
    if pos >= 0:
        return text[pos:]
    pos = text.find('onnxcustom')
    if pos >= 0:
        return text[pos:]
    pos = text.find('site-packages')
    if pos >= 0:
        return text[pos:]
    return text


ps = profile(lambda: benchmark(X_train, y_train,
             nn, train_session, name='NN-CPU'))[0]
root, nodes = profile2graph(ps, clean_text=clean_name)
text = root.to_text()
print(text)

######################################
# if GPU is available
# +++++++++++++++++++

if get_device().upper() == 'GPU':

    train_session = OrtGradientForwardBackwardOptimizer(
        onx, device='cuda', learning_rate=1e-5,
        warm_start=False, max_iter=200, batch_size=batch_size)

    benches.append(benchmark(X_train, y_train, nn,
                   train_session, name='NN-GPU'))

######################################
# Linear Regression
# +++++++++++++++++

lr = MLPRegressor(hidden_layer_sizes=tuple(), max_iter=max_iter,
                  solver='sgd', learning_rate_init=5e-2, alpha=0,
                  n_iter_no_change=max_iter * 3, batch_size=batch_size,
                  nesterovs_momentum=False, momentum=0,
                  learning_rate="invscaling")

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lr.fit(X, y)


onx = to_onnx(lr, X_train[:1].astype(numpy.float32), target_opset=15)

train_session = OrtGradientForwardBackwardOptimizer(
    onx, device='cpu', learning_rate=5e-4,
    warm_start=False, max_iter=max_iter, batch_size=batch_size)

benches.append(benchmark(X_train, y_train, lr, train_session, name='LR-CPU'))

if get_device().upper() == 'GPU':

    train_session = OrtGradientForwardBackwardOptimizer(
        onx, device='cuda', learning_rate=5e-4,
        warm_start=False, max_iter=200, batch_size=batch_size)

    benches.append(benchmark(X_train, y_train, nn,
                   train_session, name='LR-GPU'))


######################################
# GPU profiling
# +++++++++++++

if get_device().upper() == 'GPU':
    ps = profile(lambda: benchmark(X_train, y_train,
                 lr, train_session, name='LR-GPU'))[0]
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

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df[['skl', 'ort']].plot.bar(title="Processing time", ax=ax[0])
ax[0].tick_params(axis='x', rotation=30)
for bench in benches:
    ax[1].plot(bench['losses_skl'][1:], label='skl-' + bench['name'])
    ax[1].plot(bench['losses_ort'][1:], label='ort-' + bench['name'])
ax[1].set_yscale('log')
ax[1].set_title("Losses")
ax[1].legend()

########################################
# The gradient update are not exactly the same.
# It should be improved for a fair comprison.

# plt.show()
