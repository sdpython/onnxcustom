"""

.. _l-orttraining-linreg-fwbw:

Train a linear regression with forward backward
===============================================

This example rewrites :ref:`l-orttraining-linreg` with another
optimizer :class:`OrtGradientForwardBackwardOptimizer
<onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer>`.
This optimizer relies on class :epkg:`TrainingAgent` from
:epkg:`onnxruntime-training`. In this case, the user does not have to
modify the graph to compute the error. The optimizer
builds another graph which returns the gradient of every weights
assuming the gradient on the output is known. Finally, the optimizer
adds the gradients to the weights. To summarize, it starts from the following
graph:

.. image:: images/onnxfwbw1.png

Class :class:`OrtGradientForwardBackwardOptimizer
<onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer>`
builds other ONNX graph to implement a gradient descent algorithm:

.. image:: images/onnxfwbw2.png

The blue node is built by class :epkg:`TrainingAgent`
(from :epkg:`onnxruntime-training`). The green nodes are added by
class :class:`OrtGradientForwardBackwardOptimizer
<onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer>`.
This implementation relies on ONNX to do the computation but it could
be replaced by any other framework such as :epkg:`pytorch`. This
design gives more freedom to the user to implement his own training
algorithm.

.. contents::
    :local:

A simple linear regression with scikit-learn
++++++++++++++++++++++++++++++++++++++++++++

"""
from pprint import pprint
import numpy
from pandas import DataFrame
from onnxruntime import get_device
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from mlprodict.onnx_conv import to_onnx
from onnxcustom.plotting.plotting_onnx import plot_onnxs
from onnxcustom.utils.orttraining_helper import get_train_initializer
from onnxcustom.training.optimizers_partial import (
    OrtGradientForwardBackwardOptimizer)

X, y = make_regression(n_features=2, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

############################################
# We use a :class:`sklearn.neural_network.MLPRegressor`.

lr = MLPRegressor(hidden_layer_sizes=tuple(),
                  activation='identity', max_iter=50,
                  batch_size=10, solver='sgd',
                  alpha=0, learning_rate_init=1e-2,
                  n_iter_no_change=200,
                  momentum=0, nesterovs_momentum=False)
lr.fit(X, y)
print(lr.predict(X[:5]))

##################################
# The trained coefficients are:
print("trained coefficients:", lr.coefs_, lr.intercepts_)

##################################
# ONNX graph
# ++++++++++
#
# Training with :epkg:`onnxruntime-training` starts with an ONNX
# graph which defines the model to learn. It is obtained by simply
# converting the previous linear regression into ONNX.

onx = to_onnx(lr, X_train[:1].astype(numpy.float32), target_opset=15,
              black_op={'LinearRegressor'})

plot_onnxs(onx, title="Linear Regression in ONNX")

#####################################
# Weights
# +++++++
#
# Every initializer is a set of weights which can be trained
# and a gradient will be computed for it.
# However an initializer used to modify a shape or to
# extract a subpart of a tensor does not need training.
# :func:`get_train_initializer
# <onnxcustom.tools.orttraining_helper.get_train_initializer>`
# removes them.

inits = get_train_initializer(onx)
weights = {k: v for k, v in inits.items() if k != "shape_tensor"}
pprint(list((k, v[0].shape) for k, v in weights.items()))

#####################################
# Train on CPU or GPU if available
# ++++++++++++++++++++++++++++++++

device = "cuda" if get_device().upper() == 'GPU' else 'cpu'
print("device=%r get_device()=%r" % (device, get_device()))

#######################################
# Stochastic Gradient Descent
# +++++++++++++++++++++++++++
#
# The training logic is hidden in class
# :class:`OrtGradientForwardBackwardOptimizer
# <onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer>`
# It follows :epkg:`scikit-learn` API (see `SGDRegressor
# <https://scikit-learn.org/stable/modules/
# generated/sklearn.linear_model.SGDRegressor.html>`_.

train_session = OrtGradientForwardBackwardOptimizer(
    onx, list(weights), device=device, verbose=1, learning_rate=1e-2,
    warm_start=False, max_iter=200, batch_size=10)

train_session.fit(X, y)

######################################
# And the trained coefficients are...

state_tensors = train_session.get_state()
pprint(["trained coefficients:", state_tensors])
print("last_losses:", train_session.train_losses_[-5:])

min_length = min(len(train_session.train_losses_), len(lr.loss_curve_))
df = DataFrame({'ort losses': train_session.train_losses_[:min_length],
                'skl losses': lr.loss_curve_[:min_length]})
df.plot(title="Train loss against iterations")

#######################################
# The convergence speed is almost the same.
#
# Gradient Graph
# ++++++++++++++
#
# As mentioned in this introduction, the computation relies
# on a few more graphs than the initial graph.
# When the loss is needed but not the gradient, class
# :epkg:`TrainingAgent` creates another graph, faster,
# with the trained initializers as additional inputs.

onx_loss = train_session.train_session_.cls_type_._optimized_pre_grad_model

plot_onnxs(onx, onx_loss, title=['regression', 'loss'])

#####################################
# And the gradient.

onx_gradient = train_session.train_session_.cls_type_._trained_onnx

plot_onnxs(onx_loss, onx_gradient, title=['loss', 'gradient + loss'])

#################################
# The last ONNX graphs are used to compute the gradient *dE/dY*
# and to update the weights. The first graph takes the labels and the
# expected labels and returns the square loss and its gradient.
# The second graph takes the weights and the learning rate as inputs
# and returns the updated weights. This graph works on tensors of any shape
# but with the same element type.

plot_onnxs(train_session.learning_loss.loss_grad_onnx_,
           train_session.learning_rate.axpy_onnx_,
           title=['error gradient + loss', 'gradient update'])

# import matplotlib.pyplot as plt
# plt.show()
