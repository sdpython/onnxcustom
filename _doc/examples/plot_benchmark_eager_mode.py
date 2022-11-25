"""
.. _benchmark-ort-eager-mode:

Benchmark onnxruntime API: eager mode
=====================================

epkg:`pytorch` or :epkg:`tensorflow` usually work faster if the
deep learning model is entirely run outside python. The python code
is only used to build the model but is then used to call the
execution of the whole. In that configuration, there is no way
to look into intermediate results.

It does not make it easy to debug or investigate what is going on.
What the user writes is not what is executed.
Eager mode is an expression which defines a situation where
the code which defines the model is the same as the used to
execute the model. Everything happens in python. It is slower
but the gap is small if the model manipulate big matrices.

It is possible to do the same with :epkg:`onnxruntime`.
This example compares the performance of a couple of
scenarios. This work is close to what is done in example
:ref:`benchmark-ort-api`.

.. contents::
    :local:

The scenario
++++++++++++

We would like to compare two codes. The first one
executes 2 additions in a single onnx graph. The second
one executes 10 additions, each of them calling :epkg:`onnxruntime`
for a single addition.

"""
import time
import numpy
from numpy.testing import assert_allclose
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from onnx import TensorProto
from onnx.numpy_helper import from_array
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnxruntime import InferenceSession, __version__ as ort_version
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    SessionIOBinding, OrtDevice as C_OrtDevice,
    OrtMemType, OrtValue as C_OrtValue, RunOptions)
from cpyquickhelper.numbers.speed_measure import measure_time
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation

############################################
# Available optimisation on this machine.

print(code_optimisation())
repeat = 250
number = 250

############################################
# A single addition of a matrix of two dimension.

CST = numpy.array(list(range(100))).reshape(1, -1).astype(numpy.float32)
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, CST.shape[1]])
Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None, CST.shape[1]])

graph = make_graph([
    make_node("Add", ['X', 'Y'], ['Z']),
], '', [X], [Z], [
    from_array(CST, name='Y'),
])
onnx_add = make_model(graph)
sess_add = InferenceSession(onnx_add.SerializeToString(),
                            providers=["CPUExecutionProvider"])

#############################################
# Two additions of the same matrix.

graph = make_graph([
    make_node("Add", ['X', 'Y'], ['T']),
    make_node("Add", ['T', 'Y'], ['Z']),
], '', [X], [Z], [
    from_array(CST, 'Y'),
])
onnx_add2 = make_model(graph)
sess_add2 = InferenceSession(onnx_add2.SerializeToString(),
                             providers=["CPUExecutionProvider"])

############################################
# The functions to test
# +++++++++++++++++++++
#
# * `numpy`: :epkg:`numpy`
# * `ort`: :epkg:`onnxruntime` + numpy array as input
# * `ort-ov`: :epkg:`onnxruntime` + :epkg:`C_OrtValue` as input


def f_numpy(X):
    "numpy"
    T = X + CST
    Z = T + CST
    return Z


def f_ort_eager(X):
    "ort-eager"
    T = sess_add._sess.run(['Z'], {'X': X}, None)[0]
    Z = sess_add._sess.run(['Z'], {'X': T}, None)[0]
    return Z


def f_ort(X):
    "ort"
    Z = sess_add2._sess.run(['Z'], {'X': X}, None)[0]
    return Z


def f_ort_ov_eager(X):
    "ort-ov-eager"
    T = sess_add._sess.run_with_ort_values({'X': X}, ['Z'], None)[0]
    Z = sess_add._sess.run_with_ort_values({'X': T}, ['Z'], None)[0]
    return Z


def f_ort_ov(X):
    "ort-ov"
    Z = sess_add2._sess.run_with_ort_values({'X': X}, ['Z'], None)[0]
    return Z


X = numpy.random.rand(10, CST.shape[1]).astype(CST.dtype)

device = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
Xov = C_OrtValue.ortvalue_from_numpy(X, device)

Ys = [
    f_numpy(X),
    f_ort_eager(X),
    f_ort(X),
    f_ort_ov_eager(Xov),
    f_ort_ov(Xov),
]

for i in range(1, len(Ys)):
    try:
        assert_allclose(Ys[0], Ys[i])
    except TypeError:
        # OrtValue
        assert_allclose(Ys[0], Ys[i].numpy())

##########################################
# All outputs are the same.

##############################
# Benchmark the functions
# +++++++++++++++++++++++


def benchmark(repeat=100):
    fcts = [f_numpy, f_ort_eager, f_ort, f_ort_ov_eager, f_ort_ov]
    data = []
    for N in tqdm([1, 2, 5, 10, 20, 50, 100, 200, 500,
                   1000, 2000, 5000, 10000, 20000]):
        X = numpy.random.rand(N, CST.shape[1]).astype(CST.dtype)
        device = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
        Xov = C_OrtValue.ortvalue_from_numpy(X, device)

        for f in fcts:
            obs = {'name': f.__doc__, "N": N}
            if "-ov" in f.__doc__:
                begin = time.perf_counter()
                for r in range(repeat):
                    _ = f(Xov)
                end = time.perf_counter() - begin
            else:
                begin = time.perf_counter()
                for r in range(repeat):
                    _ = f(X)
                end = time.perf_counter() - begin
            obs['time'] = end / repeat
            data.append(obs)

    return pandas.DataFrame(data)


df = benchmark()
df


########################################
# Graphs
# ++++++

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

piv = df.pivot(index="N", columns="name", values="time")
piv.plot(ax=ax[0], title="Time(s) per execution", logy=True, logx=True)
piv2 = piv / piv.index.values.reshape((-1, 1))
piv2.plot(ax=ax[1], title="Time(s) per execution / N", logx=True)
piv3 = piv / piv["numpy"].values.reshape((-1, 1))
piv3.plot(ax=ax[2], title="Ratio against numpy (lower is better)",
          logy=True, logx=True)


###################################
# Conclusion
# ++++++++++
#
# The eager mode is slower than numpy for small arrays than is faster.
# This is probably due to :epkg:`pybind11` binding when numpy
# is using the direct python API. This could be improved by using :epkg:`cython`.
# Eager mode must use :epkg:`OrtValue`. It is faster and it reduces the differences
# between using two additions in a single graph or two graphs of a single addition.

print(f"onnxruntime.__version__ = {ort_version!r}")


plt.show()
