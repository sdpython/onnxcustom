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
:ref:`benchmark-ort-api`. The example compares the performance
of a couple of methods for CPU and GPU.

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
    make_model, make_node,
    make_graph, make_tensor_value_info)
from onnxruntime import (
    get_all_providers, InferenceSession, __version__ as ort_version,
    RunOptions)
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtDevice as C_OrtDevice,
    OrtMemType, OrtValue as C_OrtValue)
try:
    from onnxruntime.capi._pybind_state import OrtValueVector
except ImportError:
    # You need onnxruntime>=1.14
    OrtValueVector = None
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
# Let's consider GPU as well.

has_cuda = "CUDAExecutionProvider" in get_all_providers()
if has_cuda:
    sess_add_gpu = InferenceSession(onnx_add.SerializeToString(),
                                    providers=["CUDAExecutionProvider"])
    sess_add2_gpu = InferenceSession(onnx_add2.SerializeToString(),
                                     providers=["CUDAExecutionProvider"])
else:
    print("No GPU or one GPU was detected.")
    sess_add_gpu = None
    sess_add2_gpu = None

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

#######################################
# onnxruntime >= 1.14 introduces a vector of OrtValues
# to bypass the building of a dictionary.


if OrtValueVector is not None:

    vect_out = OrtValueVector()
    run_options = RunOptions()
    devices = [C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)]

    def f_ort_vect_ov_eager(X):
        "ort-vect-ov-eager"
        vect_in = OrtValueVector()
        vect_in.push_back(X)
        temp_vect_out = OrtValueVector()
        sess_add._sess.run_with_ortvaluevector(
            run_options, ["X"], vect_in, ["Z"], temp_vect_out, devices)
        assert len(temp_vect_out) == 1
        sess_add._sess.run_with_ortvaluevector(
            run_options, ["X"], temp_vect_out, ["Z"], vect_out, devices)
        assert len(vect_out) == 1
        return vect_out[0]

    def f_ort_vect_ov(X):
        "ort-vect-ov"
        vect_in = OrtValueVector()
        vect_in.push_back(X)
        sess_add2._sess.run_with_ortvaluevector(
            run_options, ["X"], vect_in, ["Z"], vect_out, devices)
        assert len(vect_out) == 1
        return vect_out[0]

else:
    f_ort_vect_ov_eager = None
    f_ort_vect_ov = None

#########################################
# If GPU is available.

if sess_add_gpu is not None:
    #

    def f_ort_ov_eager_gpu(X):
        "ort-ov-eager-gpu"
        T = sess_add_gpu._sess.run_with_ort_values({'X': X}, ['Z'], None)[0]
        Z = sess_add_gpu._sess.run_with_ort_values({'X': T}, ['Z'], None)[0]
        return Z

    def f_ort_ov_gpu(X):
        "ort-ov-gpu"
        Z = sess_add2_gpu._sess.run_with_ort_values({'X': X}, ['Z'], None)[0]
        return Z

    if OrtValueVector is not None:

        run_options = RunOptions()
        devices = [C_OrtDevice(C_OrtDevice.cuda(), OrtMemType.DEFAULT, 0)]

        def f_ort_vect_ov_eager_gpu(X):
            "ort-vect-ov-eager-gpu"
            vect_in = OrtValueVector()
            vect_in.push_back(X)
            temp_vect_out = OrtValueVector()
            sess_add._sess.run_with_ortvaluevector(
                run_options, ["X"], vect_in, ["Z"], temp_vect_out, devices)
            sess_add._sess.run_with_ortvaluevector(
                run_options, ["X"], temp_vect_out, ["Z"], vect_out, devices)
            assert len(vect_out) == 1
            return vect_out[0]

        def f_ort_vect_ov_gpu(X):
            "ort-vect-ov-gpu"
            vect_in = OrtValueVector()
            vect_in.push_back(X)
            sess_add2._sess.run_with_ortvaluevector(
                run_options, ["X"], vect_in, ["Z"], vect_out, devices)
            assert len(vect_out) == 1
            return vect_out[0]

    else:
        f_ort_vect_ov_eager_gpu = None
        f_ort_vect_ov_gpu = None

else:
    f_ort_ov_eager_gpu = None
    f_ort_ov_gpu = None
    f_ort_vect_ov_eager_gpu = None
    f_ort_vect_ov_gpu = None


#######################################
# Let's now check all these functions produces the same results.

X = numpy.random.rand(10, CST.shape[1]).astype(CST.dtype)

device = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
Xov = C_OrtValue.ortvalue_from_numpy(X, device)

Ys = [
    (f_numpy, X),
    (f_ort_eager, X),
    (f_ort, X),
    (f_ort_ov_eager, Xov),
    (f_ort_ov, Xov),
]

if OrtValueVector is not None:
    Ys.extend([
        (f_ort_vect_ov_eager, Xov),
        (f_ort_vect_ov, Xov),
    ])

if sess_add_gpu is not None:
    device_gpu = C_OrtDevice(C_OrtDevice.cuda(), OrtMemType.DEFAULT, 0)
    try:
        Xov_gpu = C_OrtValue.ortvalue_from_numpy(X, device_gpu)
        Ys.extend([
            (f_ort_ov_eager_gpu, Xov_gpu),
            (f_ort_ov_gpu, Xov_gpu),
        ])
        if OrtValueVector is not None:
            Ys.extend([
                (f_ort_vect_ov_eager_gpu, Xov_gpu),
                (f_ort_vect_ov_gpu, Xov_gpu),
            ])
    except RuntimeError:
        # cuda is not available
        sess_add_gpu = None
        sess_add2_gpu
        f_ort_ov_eager_gpu = None
        f_ort_ov_gpu = None

results = []
for fct, x in Ys:
    print(f"check function {fct.__name__!r} and input type {x.__class__.__name__!r}")
    results.append(fct(x))

for i in range(1, len(results)):
    try:
        assert_allclose(results[0], results[i])
    except TypeError:
        # OrtValue
        assert_allclose(results[0], results[i].numpy())

##########################################
# All outputs are the same.

##############################
# Benchmark the functions
# +++++++++++++++++++++++


def benchmark(repeat=100):
    fcts = [f_numpy, f_ort_eager, f_ort, f_ort_ov_eager, f_ort_ov,
            f_ort_vect_ov_eager, f_ort_vect_ov,
            f_ort_ov_eager_gpu, f_ort_ov_gpu,
            f_ort_vect_ov_gpu, f_ort_vect_ov_eager_gpu]
    data = []
    for N in tqdm([1, 2, 5, 10, 20, 50, 100, 200, 500,
                   1000, 2000, 5000, 10000, 20000]):
        X = numpy.random.rand(N, CST.shape[1]).astype(CST.dtype)
        device = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
        Xov = C_OrtValue.ortvalue_from_numpy(X, device)
        if f_ort_ov_gpu is not None:
            device_gpu = C_OrtDevice(C_OrtDevice.cuda(), OrtMemType.DEFAULT, 0)
            Xov_gpu = C_OrtValue.ortvalue_from_numpy(X, device_gpu)

        for f in fcts:
            if f is None:
                continue
            print(N, f)
            obs = {'name': f.__doc__, "N": N}
            if "-gpu" in f.__doc__:
                begin = time.perf_counter()
                for r in range(repeat):
                    _ = f(Xov_gpu)
                end = time.perf_counter() - begin
            elif "-ov" in f.__doc__:
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
df.to_csv("plot_benchmark_eager_mode.csv", index=False)
df


########################################
# Graphs
# ++++++

def make_graph(df):

    def subgraph(row, cols, title):
        if "numpy" not in cols:
            cols.append("numpy")
        piv = piv_all[cols].copy()
        piv.plot(ax=ax[row, 0], title=title, logy=True, logx=True)
        piv2 = piv / piv.index.values.reshape((-1, 1))
        piv2.plot(ax=ax[row, 1], title=f"Time(s) per execution / N", logx=True)
        piv3 = piv / piv["numpy"].values.reshape((-1, 1))
        piv3.plot(ax=ax[row, 2], title="Ratio against numpy (lower is better",
                  logy=True, logx=True)
        for j in range(0, 3):
            ax[row, j].legend(fontsize="x-small")

    fig, ax = plt.subplots(3, 3, figsize=(12, 8))
    fig.suptitle("Time execution Eager Add + Add")

    piv_all = df.pivot(index="N", columns="name", values="time")

    # no gpu, no vect
    subgraph(0, [c for c in piv_all.columns
                 if "-gpu" not in c and "-vect" not in c],
             title="CPU")

    # no gpu
    subgraph(1, [c for c in piv_all.columns
                 if "-gpu" not in c and "-ov" in c],
             title="CPU, OrtValue and OrtValueVector")

    # gpu
    cols = [c for c in piv_all.columns if "-gpu" in c and "-ov" in c]
    subgraph(2, cols,
             title="GPU, OrtValue and OrtValueVector")
    fig.savefig("eager_mode_cpu.png" if len(cols) == 0
                else "eager_mode_gpu.png", dpi=250)
    return fig, ax


fig, ax = make_graph(df)

###################################
# Conclusion
# ++++++++++
#
# The eager mode is slower than numpy for small arrays then is faster.
# This is probably due to :epkg:`pybind11` binding when numpy
# is using the direct python API. This could be improved by using :epkg:`cython`.
# Eager mode must use :epkg:`OrtValue`. It is faster and it reduces the differences
# between using two additions in a single graph or two graphs of a single addition
# on CPU. On GPU, it is still faster but eager mode is significantly slower.

if not has_cuda:
    print("With GPU")
    df = pandas.read_csv("data/eager_mode.csv")
    _, ax = make_graph(df)
else:
    ax = None
ax

######################################
# Results obtained with the following version.

print(f"onnxruntime.__version__ = {ort_version!r}")

# plt.show()
