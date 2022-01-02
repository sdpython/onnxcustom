"""

.. _l-orttraining-nn-gpu-fwbw:

Forward backward on a neural network on GPU
===========================================

This example leverages example :ref:`l-orttraining-linreg-gpu` to
train a neural network from :epkg:`scikit-learn` on GPU. The code
uses the same code introduced in :ref:`l-orttraining-linreg-fwbw`.

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
from onnxcustom.utils.orttraining_helper import get_train_initializer
from onnxcustom.utils.onnx_helper import onnx_rename_weights
from onnxcustom.training.optimizers_partial import (
    OrtGradientForwardBackwardOptimizer)


X, y = make_regression(1000, n_features=10, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

nn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=100,
                  solver='sgd', learning_rate_init=5e-5,
                  n_iter_no_change=1000, batch_size=10, alpha=0,
                  momentum=0, nesterovs_momentum=False)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nn.fit(X_train, y_train)

print(nn.loss_curve_)

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
device = "cuda" if get_device().upper() == 'GPU' else 'cpu'

print("device=%r get_device()=%r" % (device, get_device()))

#######################################
# The training session. The first instructions fails
# for an odd reason as the class :epkg:`TrainingAgent` expects
# to find the list of weights to train in alphabetical order.
# That means the list `onx.graph.initializer` must be sorted
# by alphabetical order of their names otherwise the process
# could crash unless it is caught earlier with the following
# exception.

try:
    train_session = OrtGradientForwardBackwardOptimizer(
        onx, device=device, verbose=1,
        warm_start=False, max_iter=100, batch_size=10)
    train_session.fit(X, y)
except ValueError as e:
    print(e)

#########################################
# Function :func:`onnx_rename_weights
# <onnxcustom.utils.onnx_helper.onnx_rename_weights>`
# does not change the order of the initializer but renames
# them. Then class :epkg:`TrainingAgent` may work.

onx = onnx_rename_weights(onx)
train_session = OrtGradientForwardBackwardOptimizer(
    onx, device=device, verbose=1,
    learning_rate=5e-5, warm_start=False, max_iter=100, batch_size=10)
train_session.fit(X, y)

#########################################
# Let's see the weights.

state_tensors = train_session.get_state()

##########################################
# And the loss.

print(train_session.train_losses_)

df = DataFrame({'ort losses': train_session.train_losses_,
                'skl losses:': nn.loss_curve_})
df.plot(title="Train loss against iterations", logy=True)

##############################################
# The convergence rate is different but both classes
# do not update the learning the same way.

# import matplotlib.pyplot as plt
# plt.show()
