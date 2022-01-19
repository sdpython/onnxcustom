"""
.. _l-example-discrepencies-sigmoid:

Funny discrepancies
===================

Function sigmoid is :math:`sig(x) = \\frac{1}{1 + e^{-x}}`.
For small or high value, implementation has to do approximation
and they are not always the same. It may be a tradeoff between
precision and computation time...
It is always a tradeoff.

.. index:: discrepencies, sigmoid

.. contents::
    :local:


Precision
+++++++++

This section compares the precision of a couple of implementations
of the ssigmoid function. The custom implementation is done with
a Taylor expansion of exponential function:
:math:`e^x \\sim 1 + x + \\frac{x^2}{2} + ... + \\frac{x^n}{n!} + o(x^n)`.

"""
import time
import numpy
import pandas
from tqdm import tqdm
from scipy.special import expit

from skl2onnx.algebra.onnx_ops import OnnxSigmoid
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime import InferenceSession
from mlprodict.onnxrt import OnnxInference
from onnxcustom import get_max_opset
import matplotlib.pyplot as plt

one = numpy.array([1], dtype=numpy.float64)


def taylor_approximation_exp(x, degre=50):
    y = numpy.zeros(x.shape, dtype=x.dtype)
    a = numpy.ones(x.shape, dtype=x.dtype)
    for i in range(1, degre + 1):
        a *= x / i
        y += a
    return y


def taylor_sigmoid(x, degre=50):
    den = one + taylor_approximation_exp(-x, degre)
    return one / (den)


opset = get_max_opset()
N = 300
min_values = [-20 + float(i) * 10 / N for i in range(N)]
data = numpy.array([0], dtype=numpy.float32)

node = OnnxSigmoid('X', op_version=opset, output_names=['Y'])
onx = node.to_onnx({'X': FloatTensorType()},
                   {'Y': FloatTensorType()},
                   target_opset=opset)
rts = ['numpy', 'python', 'onnxruntime', 'taylor20', 'taylor40']

oinf = OnnxInference(onx)
sess = InferenceSession(onx.SerializeToString())

graph = []
for mv in tqdm(min_values):
    data[0] = mv
    for rt in rts:
        lab = ""
        if rt == 'numpy':
            y = expit(data)
        elif rt == 'python':
            y = oinf.run({'X': data})['Y']
            # * 1.2 to avoid curves to be superimposed
            y *= 1.2
            lab = "x1.2"
        elif rt == 'onnxruntime':
            y = sess.run(None, {'X': data})[0]
        elif rt == 'taylor40':
            y = taylor_sigmoid(data, 40)
            # * 0.8 to avoid curves to be superimposed
            y *= 0.8
            lab = "x0.8"
        elif rt == 'taylor20':
            y = taylor_sigmoid(data, 20)
            # * 0.6 to avoid curves to be superimposed
            y *= 0.6
            lab = "x0.6"
        else:
            raise AssertionError("Unknown runtime %r." % rt)
        value = y[0]
        graph.append(dict(rt=rt + lab, x=mv, y=value))

#############################################
# Graph.

_, ax = plt.subplots(1, 1, figsize=(12, 4))
df = pandas.DataFrame(graph)
piv = df.pivot('x', 'rt', 'y')
print(piv.T.head())
piv.plot(ax=ax, logy=True)

##############################################
# :math:`log(sig(x)) = -log(1 + e^{-x})`. When *x* is very negative,
# :math:`log(sig(x)) \\sim -x`. That explains the graph.
# We also see :epkg:`onnxruntime` is less precise for these values.
# What's the benefit?
#
# Computation time
# ++++++++++++++++

graph = []
for mv in tqdm(min_values):
    data = numpy.array([mv] * 10000, dtype=numpy.float32)
    for rt in rts:
        begin = time.perf_counter()
        if rt == 'numpy':
            y = expit(data)
        elif rt == 'python':
            y = oinf.run({'X': data})['Y']
        elif rt == 'onnxruntime':
            y = sess.run(None, {'X': data})[0]
        elif rt == 'taylor40':
            y = taylor_sigmoid(data, 40)
        elif rt == 'taylor20':
            y = taylor_sigmoid(data, 20)
        else:
            raise AssertionError("Unknown runtime %r." % rt)
        duration = time.perf_counter() - begin
        graph.append(dict(rt=rt, x=mv, y=duration))

_, ax = plt.subplots(1, 1, figsize=(12, 4))
df = pandas.DataFrame(graph)
piv = df.pivot('x', 'rt', 'y')
piv.plot(ax=ax, logy=True)

#############################################
# Conclusion
# ++++++++++
#
# The implementation from :epkg:`onnxruntime` is faster but
# is much less contiguous for extremes. It explains why
# probabilities may be much different when an observation
# is far from every classification border. In that case,
# :epkg:`onnxruntime` implementation of the sigmoid function
# returns 0 when :func:`numpy.sigmoid` returns a smoother value.
# Probabilites of logistic regression are obtained after the raw
# scores are transformed with the sigmoid function and
# normalized. If the raw scores are very negative,
# the sum of probabilities becomes null with :epkg:`onnxruntime`.
# The normalization fails.

# plt.show()
