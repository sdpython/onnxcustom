"""

.. _l-orttraining-nn-gpu:

Train a scikit-learn neural network with onnxruntime-training on GPU
====================================================================

This example leverages example :ref:`l-orttraining-linreg-gpu` to
train a neural network from :epkg:`scikit-learn` on GPU.

.. contents::
    :local:

A neural network with scikit-learn
++++++++++++++++++++++++++++++++++

"""
import warnings
from pprint import pprint
import numpy
from pandas import DataFrame
from onnx import helper, numpy_helper, TensorProto
from onnxruntime import (
    __version__ as ort_version, get_device, OrtValue,
    TrainingParameters, SessionOptions, TrainingSession,
    InferenceSession)
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from mlprodict.plotting.plotting_onnx import plot_onnx
from mlprodict.onnx_conv import to_onnx
from onnxcustom.training import add_loss_output, get_train_initializer
from onnxcustom.training.data_loader import OrtDataLoader
from onnxcustom.training.optimizers import OrtGradientOptimizer
from tqdm import tqdm

X, y = make_regression(1000, n_features=10, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

nn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000)

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
plot_onnx(onx)

#######################################
# Training graph
# ++++++++++++++
#
# The loss function is the square function. We use function
# :func:`add_loss_output <onnxcustom.training.orttraining.add_loss_output>`.
# It does something what is implemented in example
# :ref:`l-orttraining-linreg-cpu`.

onx_train = add_loss_output(onx)
plot_onnx(onx_train)

#####################################
# Let's check inference is working.

sess = InferenceSession(onx_train.SerializeToString())
res = sess.run(None, {'X': X_test, 'label': y_test.reshape((-1, 1))})
print("onnx loss=%r" % (res[0][0, 0] / X_test.shape[0]))

#####################################
# Let's retrieve the constant, the weight to optimize.
# We remove initializer which cannot be optimized.

inits = get_train_initializer(onx)
weights = {k: v for k, v in inits.items() if k != "shape_tensor"}
pprint(list((k, v[0].shape) for k, v in weights.items()))


######################################
# Training
# ++++++++
#
# The training session. If GPU is available, it chooses CUDA
# otherwise it falls back to CPU.

device = "cuda" if get_device() == 'GPU' else 'cpu'

print("device=%r get_device()=%r" % (device, get_device()))

#######################################
# The training session.

train_session = OrtGradientOptimizer(
    onx_train, list(weights), device=device, verbose=1, eta0=1e-4)

train_session.fit(X, y)
state_tensors = train_session.get_state()

print(train_session.train_losses_)
