"""

.. _l-orttraining-nn-gpu-fwbw:

Forward backward on a neural network with onnxruntime-training on GPU
=====================================================================

This example leverages example :ref:`l-orttraining-linreg-gpu` to
train a neural network from :epkg:`scikit-learn` on GPU. However, the code
is using classes implemented in this module, following the pattern
introduced in exemple :ref:`l-orttraining-linreg`.

.. contents::
    :local:

A neural network with scikit-learn
++++++++++++++++++++++++++++++++++

"""
import warnings
import numpy
from pandas import DataFrame
from onnxruntime import get_device
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from onnxcustom.plotting.plotting_onnx import plot_onnxs
from mlprodict.onnx_conv import to_onnx
from onnxcustom.utils.onnx_orttraining import get_train_initializer
from onnxcustom.training.optimizers_partial import (
    OrtGradientForwardBackwardOptimizer)


X, y = make_regression(1000, n_features=10, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

nn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=200,
                  solver='sgd', learning_rate_init=1e-4,
                  n_iter_no_change=1000, batch_size=10)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nn.fit(X_train, y_train)

#################################
# Score:

print("mean_squared_error=%r" % mean_squared_error(y_test, nn.predict(X_test)))


#######################################
# Conversion to ONNX
# ++++++++++++++++++

onx = to_onnx(nn, X_train[:1].astype(numpy.float32), target_opset=15)
plot_onnxs(onx)

###############################
# Initializers to train

weights = list(sorted(get_train_initializer(onx)))
print(weights)

#######################################
# Training graph with forward backward
# ++++++++++++++++++++++++++++++++++++
#
device = "cuda" if get_device() == 'GPU' else 'cpu'

print("device=%r get_device()=%r" % (device, get_device()))

#######################################
# The training session.

train_session = OrtGradientForwardBackwardOptimizer(
    onx, list(weights), device=device, verbose=1,
    learning_rate=5e-4, warm_start=False, max_iter=200, batch_size=10)

train_session.fit(X, y)
state_tensors = train_session.get_state()

print(train_session.train_losses_)

df = DataFrame({'ort losses': train_session.train_losses_,
                'skl losses:': nn.loss_curve_})
df.plot(title="Train loss against iterations", logy=True)

# import matplotlib.pyplot as plt
# plt.show()
