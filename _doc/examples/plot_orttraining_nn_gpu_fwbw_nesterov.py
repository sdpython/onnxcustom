"""

.. _l-orttraining-nn-gpu-fwbw-nesterov:

Forward backward on a neural network on GPU (Nesterov) and penalty
==================================================================

This example does the same as :ref:`l-orttraining-nn-gpu-fwbw`
but updates the weights using `Nesterov momentum
<https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum>`_.

.. contents::
    :local:

A neural network with scikit-learn
++++++++++++++++++++++++++++++++++

"""
import warnings
import numpy
import onnx
from pandas import DataFrame
from onnxruntime import get_device
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from onnxcustom.plotting.plotting_onnx import plot_onnxs
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from onnxcustom.utils.orttraining_helper import get_train_initializer
from onnxcustom.utils.onnx_helper import onnx_rename_weights
from onnxcustom.training.optimizers_partial import (
    OrtGradientForwardBackwardOptimizer)
from onnxcustom.training.sgd_learning_rate import LearningRateSGDNesterov
from onnxcustom.training.sgd_learning_penalty import ElasticLearningPenalty


X, y = make_regression(1000, n_features=10, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

nn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=100,
                  solver='sgd', learning_rate_init=5e-5,
                  n_iter_no_change=1000, batch_size=10, alpha=0,
                  momentum=0.9, nesterovs_momentum=True)

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

weights = list(sorted(get_train_initializer(onx)))
print(weights)

#######################################
# Training graph with forward backward
# ++++++++++++++++++++++++++++++++++++
#
device = "cuda" if get_device().upper() == 'GPU' else 'cpu'

print("device=%r get_device()=%r" % (device, get_device()))

onx = onnx_rename_weights(onx)
train_session = OrtGradientForwardBackwardOptimizer(
    onx, device=device, verbose=1,
    learning_rate=LearningRateSGDNesterov(1e-4, nesterov=True, momentum=0.9),
    warm_start=False, max_iter=100, batch_size=10)
train_session.fit(X, y)

#########################################
# Let's see the weights.

state_tensors = train_session.get_state()

##########################################
# And the loss.

print(train_session.train_losses_)

df = DataFrame({'ort losses': train_session.train_losses_,
                'skl losses:': nn.loss_curve_})
df.plot(title="Train loss against iterations (Nesterov)", logy=True)

##############################################
# The convergence rate is different but both classes
# do not update the learning exactly the same way.

##############################################
# Regularization
# ++++++++++++++
#
# Default parameters for MLPRegressor suggest to penalize weights
# during training: `alpha=1e-4`.

nn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=100,
                  solver='sgd', learning_rate_init=5e-5,
                  n_iter_no_change=1000, batch_size=10, alpha=1e-4,
                  momentum=0.9, nesterovs_momentum=True)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nn.fit(X_train, y_train)

print(nn.loss_curve_)

################################################
# Let's do the same with onnxruntime.

train_session = OrtGradientForwardBackwardOptimizer(
    onx, device=device, verbose=1,
    learning_rate=LearningRateSGDNesterov(1e-4, nesterov=True, momentum=0.9),
    learning_penalty=ElasticLearningPenalty(l1=0, l2=1e-4),
    warm_start=False, max_iter=100, batch_size=10)
train_session.fit(X, y)

#########################################
# Let's see the weights.

state_tensors = train_session.get_state()

##########################################
# And the loss.

print(train_session.train_losses_)

df = DataFrame({'ort losses': train_session.train_losses_,
                'skl losses:': nn.loss_curve_})
df.plot(title="Train loss against iterations (Nesterov + penalty)", logy=True)

###########################################
# All ONNX graphs
# +++++++++++++++
#
# Method Method :meth:`save_onnx_graph
# <onnxcustom.training._base.BaseOnnxClass.save_onnx_graph>`
# can export all the ONNX graph used by the model on disk.


def print_graph(d):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            print_graph(v)
        else:
            print("\n++++++", v.replace("\\", "/"), "\n")
            with open(v, "rb") as f:
                print(onnx_simple_text_plot(onnx.load(f)))


all_files = train_session.save_onnx_graph('.')
print_graph(all_files)


# import matplotlib.pyplot as plt
# plt.show()
