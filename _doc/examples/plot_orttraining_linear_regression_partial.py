"""

.. _l-orttraining-linreg-partial:

Partial training of a linear regression with onnxruntime-training
=================================================================


.. contents::
    :local:

A simple linear regression with scikit-learn
++++++++++++++++++++++++++++++++++++++++++++

"""
from pprint import pprint
import numpy
from pandas import DataFrame
from onnxruntime import (
    InferenceSession, get_device)
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from mlprodict.onnx_conv import to_onnx
from onnxcustom.plotting.plotting_onnx import plot_onnxs

X, y = make_regression(n_features=2, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = MLPRegressor(hidden_layer_sizes=tuple(),
                  activation='identity', max_iter=200,
                  batch_size=10, solver='sgd',
                  alpha=0, learning_rate_init=1e-2,
                  n_iter_no_change=200)
lr.fit(X, y)
print(lr.predict(X[:5]))

##################################
# The trained coefficients are:
print("trained coefficients:", lr.coefs_, lr.intercepts_)

##################################
# ONNX graph
# ++++++++++
#
# Training with :pekg:`onnxruntime-training` starts with an ONNX
# graph which defines the model to learn. It is obtained by simply
# converting the previous linear regression into ONNX.

onx = to_onnx(lr, X_train[:1].astype(numpy.float32), target_opset=15)
