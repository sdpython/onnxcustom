"""

.. _l-orttraining-benchmark-fwbw-cls:

Benchmark, comparison sklearn - forward-backward - classification
=================================================================

The benchmark compares the processing time between :epkg:`scikit-learn`
and :epkg:`onnxruntime-training` on a logistic regression regression
and a neural network for classification.
It replicates the benchmark implemented in :ref:`l-orttraining-benchmark-fwbw`.

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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnx_tools.onnx_manipulations import select_model_inputs_outputs
from onnxcustom.utils.onnx_helper import onnx_rename_weights
from onnxcustom.training.optimizers_partial import (
    OrtGradientForwardBackwardOptimizer)
from onnxcustom.training.sgd_learning_rate import LearningRateSGDNesterov
from onnxcustom.training.sgd_learning_loss import NegLogLearningLoss
from onnxcustom.training.sgd_learning_penalty import ElasticLearningPenalty


X, y = make_classification(1000, n_features=100, n_classes=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.int64)
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

nn = MLPClassifier(hidden_layer_sizes=(50, 10), max_iter=max_iter,
                   solver='sgd', learning_rate_init=1e-1, alpha=1e-4,
                   n_iter_no_change=max_iter * 3, batch_size=batch_size,
                   nesterovs_momentum=True, momentum=0.9,
                   learning_rate="invscaling")

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nn.fit(X_train, y_train)

########################################
# Conversion to ONNX and trainer initialization
# It is slightly different from a regression model.
# Probabilities usually come from raw scores transformed
# through a function such as the sigmoid function.
# The gradient of the loss is computed against the raw scores
# because it is easier to compute than to let onnxruntime
# do it.

onx = to_onnx(nn, X_train[:1].astype(numpy.float32), target_opset=15,
              options={'zipmap': False})

try:
    print(onnx_simple_text_plot(onx))
except RuntimeError as e:
    print("You should upgrade mlprodict.")
    print(e)

##########################################
# Raw scores are the input of operator *Sigmoid*.

onx = select_model_inputs_outputs(
    onx, outputs=["add_result2"], infer_shapes=True)
print(onnx_simple_text_plot(onx))

#########################################
# And the names are renamed to have them follow the
# alphabetical order (see :class:`OrtGradientForwardBackward
# <onnxcustom.training.ortgradient.OrtGradientForwardBackward>`).

onx = onnx_rename_weights(onx)
print(onnx_simple_text_plot(onx))

################################################
# We select the log loss (see :class:`NegLogLearningLoss
# <onnxcustom.training.sgd_learning_loss.NegLogLearningLoss>`,
# a simple regularization defined with :class:`ElasticLearningPenalty
# <onnxcustom.training.sgd_learning_penalty.ElasticLearningPenalty>`,
# and the Nesterov algorithm to update the weights with
# `LearningRateSGDNesterov
# <onnxcustom.training.sgd_learning_rate.LearningRateSGDNesterov>`.

train_session = OrtGradientForwardBackwardOptimizer(
    onx, device='cpu', warm_start=False,
    max_iter=max_iter, batch_size=batch_size,
    learning_loss=NegLogLearningLoss(),
    learning_rate=LearningRateSGDNesterov(
        1e-7, nesterov=True, momentum=0.9),
    learning_penalty=ElasticLearningPenalty(l1=0, l2=1e-4))


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
        onx, device='cuda', warm_start=False,
        max_iter=max_iter, batch_size=batch_size,
        learning_loss=NegLogLearningLoss(),
        learning_rate=LearningRateSGDNesterov(
            1e-7, nesterov=False, momentum=0.9),
        learning_penalty=ElasticLearningPenalty(l1=0, l2=1e-4))

    benches.append(benchmark(X_train, y_train, nn,
                   train_session, name='NN-GPU'))

#######################################
# A simple linear layer
# +++++++++++++++++++++

nn = MLPClassifier(hidden_layer_sizes=tuple(), max_iter=max_iter,
                   solver='sgd', learning_rate_init=1e-1, alpha=1e-4,
                   n_iter_no_change=max_iter * 3, batch_size=batch_size,
                   nesterovs_momentum=True, momentum=0.9,
                   learning_rate="invscaling", activation='identity')


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nn.fit(X_train, y_train)

onx = to_onnx(nn, X_train[:1].astype(numpy.float32), target_opset=15,
              options={'zipmap': False, 'nocl': True})
onx = select_model_inputs_outputs(
    onx, outputs=["add_result"], infer_shapes=True)
onx = onnx_rename_weights(onx)

try:
    print(onnx_simple_text_plot(onx))
except RuntimeError as e:
    print("You should upgrade mlprodict.")
    print(e)

train_session = OrtGradientForwardBackwardOptimizer(
    onx, device='cpu', warm_start=False,
    max_iter=max_iter, batch_size=batch_size,
    learning_loss=NegLogLearningLoss(),
    learning_rate=LearningRateSGDNesterov(
        1e-5, nesterov=True, momentum=0.9),
    learning_penalty=ElasticLearningPenalty(l1=0, l2=1e-4))


benches.append(benchmark(X_train, y_train, nn, train_session, name='LR-CPU'))

if get_device().upper() == 'GPU':

    train_session = OrtGradientForwardBackwardOptimizer(
        onx, device='cuda', warm_start=False,
        max_iter=max_iter, batch_size=batch_size,
        learning_loss=NegLogLearningLoss(),
        learning_rate=LearningRateSGDNesterov(
            1e-5, nesterov=False, momentum=0.9),
        learning_penalty=ElasticLearningPenalty(l1=0, l2=1e-4))

    benches.append(benchmark(X_train, y_train, nn,
                   train_session, name='LR-GPU'))


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
