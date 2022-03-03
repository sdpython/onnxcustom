"""
.. _l-example-benckmark-ort-leaky-relu:

Benchmark operator LeakyRelu
============================

The operator `LeakyRelu` is equivalent to the function:
:math:`LeayRelu(x) = \\begin{array}{l} x \\text{ if } x > 0  \\\\
\\alpha x \\text{otherwise} \\end{array}`. But it could be rewritten into
the following decomposition
:math:`LeayRelu(x) = x (\\indicatrice{x} + \\alpha (1 - \\indicatrice{x})) =
x ((1 - \\alpha) \\indicatrice{x} + \\alpha)`. Let's compare the
two implementation with onnx runtimes.

.. contents::
    :local:

The ONNX graphs for both implementations of LeakyRely
+++++++++++++++++++++++++++++++++++++++++++++++++++++

"""

import numpy
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
from pandas import DataFrame
from onnx import TensorProto
from onnxruntime import InferenceSession, get_device
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (
    OnnxLeakyRelu, OnnxSign, OnnxMul, OnnxAdd, OnnxDiv,
    OnnxGreater, OnnxCast)
from cpyquickhelper.numbers.speed_measure import measure_time
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
from mlprodict.plotting.plotting import onnx_simple_text_plot
from onnxcustom.plotting.plotting_onnx import plot_onnxs
from tqdm import tqdm


print([code_optimisation(), get_device()])

######################################
# First implementation: the operator LeayRelu.


def build_leaky_relu(alpha=0.5, target_opset=15):
    x = OnnxLeakyRelu('X', alpha=alpha, op_version=target_opset,
                      output_names=['Y'])
    return x.to_onnx({'X': FloatTensorType()},
                     outputs={'Y': FloatTensorType()},
                     target_opset=target_opset)


onx_leaky = build_leaky_relu()
print(onnx_simple_text_plot(onx_leaky))


#####################################
# Second option, the formula introduced above must adapted as
# ONNX operator Sign returns -1 if *x* is negative and not 0.

def build_leaky_relu_decomposed(alpha=0.5, target_opset=15):
    signo = OnnxSign('X', op_version=target_opset)
    sign = OnnxDiv(
        OnnxAdd(signo, numpy.array([1], dtype=numpy.float32),
                op_version=target_opset),
        numpy.array([2], dtype=numpy.float32), op_version=target_opset)
    fact = OnnxAdd(
        OnnxMul(sign, numpy.array([1 - alpha], dtype=numpy.float32),
                op_version=target_opset),
        numpy.array([alpha], dtype=numpy.float32),
        op_version=target_opset)
    x = OnnxMul('X', fact, op_version=target_opset,
                output_names=['Y'])
    return x.to_onnx({'X': FloatTensorType()},
                     outputs={'Y': FloatTensorType()},
                     target_opset=target_opset)


onx_leaky_dec = build_leaky_relu_decomposed()
print(onnx_simple_text_plot(onx_leaky_dec))

#####################################
# Third option, use of operater Greater


def build_leaky_relu_decomposed_greater(alpha=0.5, target_opset=15):
    signo = OnnxGreater('X', numpy.array([0], dtype=numpy.float32),
                        op_version=target_opset)
    sign = OnnxCast(signo, to=TensorProto.FLOAT,
                    op_version=target_opset)
    fact = OnnxAdd(
        OnnxMul(sign, numpy.array([1 - alpha], dtype=numpy.float32),
                op_version=target_opset),
        numpy.array([alpha], dtype=numpy.float32),
        op_version=target_opset)
    x = OnnxMul('X', fact, op_version=target_opset,
                output_names=['Y'])
    return x.to_onnx({'X': FloatTensorType()},
                     outputs={'Y': FloatTensorType()},
                     target_opset=target_opset)


onx_leaky_dec_greater = build_leaky_relu_decomposed_greater()
print(onnx_simple_text_plot(onx_leaky_dec_greater))

#########################################
# Visually

plot_onnxs(onx_leaky, onx_leaky_dec, onx_leaky_dec_greater,
           title=["One operator", "Decomposed\nLeakyRelu",
                  "Decomposed\nLeakyRelu Greater"])


############################################
# Check that both graph returns are equivalent
# ++++++++++++++++++++++++++++++++++++++++++++

sess1 = InferenceSession(onx_leaky.SerializeToString(),
                         providers=['CPUExecutionProvider'])
sess_dec = InferenceSession(onx_leaky_dec.SerializeToString(),
                            providers=['CPUExecutionProvider'])
sess_dec_greater = InferenceSession(onx_leaky_dec_greater.SerializeToString(),
                                    providers=['CPUExecutionProvider'])

for shape in [(1, ), (10, ), (5, 5), (7, 2, 4)]:
    rnd = numpy.random.randn(*shape).astype(numpy.float32)
    res1 = sess1.run(None, {'X': rnd})[0]
    res_dec = sess_dec.run(None, {'X': rnd})[0]
    res_dec_greater = sess_dec_greater.run(None, {'X': rnd})[0]
    assert_almost_equal(res1, res_dec)
    assert_almost_equal(res1, res_dec_greater)

###############################################
# Benchmark
# +++++++++

fcts = [('leakyrelu', sess1), ('dec', sess_dec),
        ('dec_greater', sess_dec_greater)]

N = 100
data = []
for dim in tqdm([10, 128, 256, 512, 1000, 2000]):
    for shape in [(N, dim), (dim, N)]:
        rnd = numpy.random.randn(*shape).astype(numpy.float32)
        for name, sess in fcts:
            repeat = int(4001 / dim)
            obs = measure_time(
                lambda: sess.run(None, {'X': rnd}),
                context=dict(rnd=rnd, sess=sess),
                div_by_number=True, repeat=repeat, number=200)
            obs['name'] = name
            obs['N'] = N
            obs['dim'] = dim
            obs['orient'] = shape[0] == N
            obs['shape'] = "%dx%d" % shape
            data.append(obs)

df = DataFrame(data)
df[['name', 'N', 'dim', 'average', 'deviation']]

print(df[['name', 'N', 'dim', 'average']])

#######################################
# Other to way to look at it.


def speedup(piv):
    for c in piv.columns:
        if c == 'leakyrelu':
            continue
        piv[c] = piv['leakyrelu'] / piv[c]
    piv['leakyrelu'] = 1
    return piv


piv = speedup(df.pivot('shape', 'name', 'average'))
piv

####################################
# Graph.

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
speedup(df[df.orient].pivot('dim', 'name', 'average')).plot(ax=ax[0])
ax[0].set_title("LeakyRelu speedup, shape=(%d,dim)"
                "\nThe higher the better" % N)
speedup(df[~df.orient].pivot('dim', 'name', 'average')).plot(ax=ax[1])
ax[1].set_title("LeakyRelu speedup, shape=(dim,%d)"
                "\nThe higher the better" % N)

####################################
# This kind of benchmark helps finding better implementation
# of operator runtime.

# plt.show()
