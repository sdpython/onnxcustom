"""

.. _l-orttraining-linreg:

Train a linear regression with onnxruntime-training
===================================================

This example explores how :epkg:`onnxruntime-training` can be used to
train a simple linear regression using a gradient descent.
It compares the results with those obtained by
:class:`sklearn.linear_model.SGDRegressor`

.. contents::
    :local:

A simple linear regression with scikit-learn
++++++++++++++++++++++++++++++++++++++++++++

"""
from pprint import pprint
import numpy
import onnx
from pandas import DataFrame
from onnxruntime import (
    InferenceSession, get_device)
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from mlprodict.onnx_conv import to_onnx
from onnxcustom.plotting.plotting_onnx import plot_onnxs
from onnxcustom.utils.orttraining_helper import (
    add_loss_output, get_train_initializer)
from onnxcustom.training.optimizers import OrtGradientOptimizer

X, y = make_regression(n_features=2, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = SGDRegressor(l1_ratio=0, max_iter=200, eta0=5e-2)
lr.fit(X, y)
print(lr.predict(X[:5]))

##################################
# The trained coefficients are:
print("trained coefficients:", lr.coef_, lr.intercept_)

############################################
# However this model does not show the training curve.
# We switch to a :class:`sklearn.neural_network.MLPRegressor`.

lr = MLPRegressor(hidden_layer_sizes=tuple(),
                  activation='identity', max_iter=200,
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

###############################################
# Choosing a loss
# +++++++++++++++
#
# The training requires a loss function. By default, it
# is the square function but it could be the absolute error or
# include regularization. Function
# :func:`add_loss_output
# <onnxcustom.utils.orttraining_helper.add_loss_output>`
# appends the loss function to the ONNX graph.

onx_train = add_loss_output(onx)

plot_onnxs(onx, onx_train,
           title=['Linear Regression',
                  'Linear Regression + Loss with ONNX'])

#####################################
# Let's check inference is working.

sess = InferenceSession(onx_train.SerializeToString(),
                        providers=['CPUExecutionProvider'])
res = sess.run(None, {'X': X_test, 'label': y_test.reshape((-1, 1))})
print("onnx loss=%r" % (res[0][0, 0] / X_test.shape[0]))

#####################################
# Weights
# +++++++
#
# Every initializer is a set of weights which can be trained
# and a gradient will be computed for it.
# However an initializer used to modify a shape or to
# extract a subpart of a tensor does not need training.
# Let's remove them from the list of initializer to train.

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
# :class:`OrtGradientOptimizer
# <onnxcustom.training.optimizers.OrtGradientOptimizer>`.
# It follows :epkg:`scikit-learn` API (see `SGDRegressor
# <https://scikit-learn.org/stable/modules/
# generated/sklearn.linear_model.SGDRegressor.html>`_.
# The gradient graph is not available at this stage.

train_session = OrtGradientOptimizer(
    onx_train, list(weights), device=device, verbose=1, learning_rate=1e-2,
    warm_start=False, max_iter=200, batch_size=10,
    saved_gradient="saved_gradient.onnx")

train_session.fit(X, y)

######################################
# And the trained coefficient are...

state_tensors = train_session.get_state()
pprint(["trained coefficients:", state_tensors])
print("last_losses:", train_session.train_losses_[-5:])

min_length = min(len(train_session.train_losses_), len(lr.loss_curve_))
df = DataFrame({'ort losses': train_session.train_losses_[:min_length],
                'skl losses': lr.loss_curve_[:min_length]})
df.plot(title="Train loss against iterations")

########################################
# the training graph looks like the following...

with open("saved_gradient.onnx.training.onnx", "rb") as f:
    graph = onnx.load(f)
    for inode, node in enumerate(graph.graph.node):
        if '' in node.output:
            for i in range(len(node.output)):
                if node.output[i] == "":
                    node.output[i] = "n%d-%d" % (inode, i)

plot_onnxs(graph, title='Training graph')

#######################################
# The convergence speed is not the same but both gradient descents
# do not update the gradient multiplier the same way.
# :epkg:`onnxruntime-training` does not implement any gradient descent,
# it just computes the gradient.
# That's the purpose of :class:`OrtGradientOptimizer
# <onnxcustom.training.optimizers.OrtGradientOptimizer>`. Next example
# digs into the implementation details.

# import matplotlib.pyplot as plt
# plt.show()
