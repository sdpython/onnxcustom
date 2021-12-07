"""
.. _example-ort-training-benchmark:

Benchmark operator Slice
========================

This short code compares the execution of the operator *Slice*
on CPU and GPU in three configurations.

.. contents::
    :local:

A simple example
++++++++++++++++

"""

import numpy
from numpy.testing import assert_almost_equal
from pandas import DataFrame, pivot_table
from onnxruntime import InferenceSession, get_device, OrtValue
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxSlice, OnnxAdd, OnnxMul
from cpyquickhelper.numbers.speed_measure import measure_time
from mlprodict.testing.experimental_c import code_optimisation
from mlprodict.onnxrt import OnnxInference
from mlprodict.plotting.plotting_onnx import plot_onnx
from tqdm import tqdm


print([code_optimisation(), get_device()])


###################################
# The graph to compare.

def build_ort_op(op_version=14, save=None, slices=None):  # opset=13, 14, ...
    if slices is None:
        starts = numpy.array([1, 1], dtype=numpy.int64)
        ends = numpy.array([-1, -1], dtype=numpy.int64)
        axes = None
    else:
        starts, ends = slices
        if starts[0] is None:
            indexes = [i for i in range(len(starts)) if starts[i] is not None]
            starts = numpy.array(
                [n for n in starts if n is not None], dtype=numpy.int64)
            ends = numpy.array(
                [n for n in ends if n is not None], dtype=numpy.int64)
            axes = numpy.array(indexes, dtype=numpy.int64)
        else:
            starts = numpy.array(starts, dtype=numpy.int64)
            ends = numpy.array(ends, dtype=numpy.int64)
            axes = None

    if axes is None:
        node1 = OnnxSlice('X', starts, ends, op_version=op_version)
    else:
        node1 = OnnxSlice('X', starts, ends, axes, op_version=op_version)
    node2 = OnnxAdd(node1, numpy.array([1], dtype=numpy.float32),
                    op_version=op_version)
    if axes is None:
        node3 = OnnxSlice(node2, starts, ends, op_version=op_version)
    else:
        node3 = OnnxSlice(node2, starts, ends, axes, op_version=op_version)
    node4 = OnnxMul(node3, numpy.array([2], dtype=numpy.float32),
                    op_version=op_version, output_names=['Y'])
    onx = node4.to_onnx(inputs=[('X', FloatTensorType([None, None]))],
                        target_opset=op_version)
    return onx


onx = build_ort_op()
plot_onnx(onx)

####################################
# Execution on CPU
# ++++++++++++++++

x = numpy.random.rand(50, 50).astype(numpy.float32)

oinf = OnnxInference(onx)
oinf.run({'X': x}, verbose=1, fLOG=print)

#################################
# With onnxruntime.

sess = InferenceSession(onx.SerializeToString(),
                        providers=["CPUExecutionProvider"])
y_cpu = sess.run(None, {'X': x})[0]


#######################################
# Execution on GPU
# ++++++++++++++++
#
# If available...

if get_device() == 'GPU':
    try:
        gx = OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
        cuda = True
    except RuntimeError as e:
        print(e)
        cuda = False
else:
    cuda = False

if cuda:
    sessg = InferenceSession(onx.SerializeToString(),
                             providers=["CUDAExecutionProvider"])

    io_binding = sessg.io_binding()
    io_binding.bind_input(
        name='X', device_type=gx.device_name(), device_id=0,
        element_type=numpy.float32, shape=gx.shape(),
        buffer_ptr=gx.data_ptr())
    io_binding.bind_output('Y')
    sessg.run_with_iobinding(io_binding)
    y_gpu = io_binding.copy_outputs_to_cpu()[0]
    assert_almost_equal(y_cpu, y_gpu)


######################################
# Benchmark
# +++++++++

data = []
shapes = ([(n, n) for n in [10, 100, 1000]] +
          [(n, 100) for n in [10, 100, 1000, 10000]] +
          [(100, n) for n in [10, 100, 1000, 10000]])
slices = [([1, 1], [-1, -1]), ([1], [-1]), ([None, 1], [None, -1])]
shape_slices = [(sh, sl) for sh in shapes for sl in slices]

for shape, slices in tqdm(shape_slices):
    onx = build_ort_op(slices=slices)
    x = numpy.random.rand(*shape).astype(numpy.float32)

    number = 100
    if x.size >= 100000:
        number = 10

    sess = InferenceSession(
        onx.SerializeToString(),
        providers=["CPUExecutionProvider"])
    sess.run(None, {'X': x})

    obs = dict(
        shape=str(shape).replace(
            " ", ""), slice=str(slices).replace(
            " ", ""))
    r = measure_time('sess.run(None, dx)', number=number, div_by_number=True,
                     context={'sess': sess, 'dx': {'X': x}})
    obs.update(r)
    obs['provider'] = 'CPU'
    data.append(obs)

    if cuda:
        def sess_run(sess, x):
            io_binding = sess.io_binding()
            io_binding.bind_input(
                name='X', device_type=gx.device_name(), device_id=0,
                element_type=numpy.float32, shape=gx.shape(),
                buffer_ptr=gx.data_ptr())
            io_binding.bind_output('Y')
            sess.run_with_iobinding(io_binding)

        sess = InferenceSession(
            onx.SerializeToString(),
            providers=["CUDAExecutionProvider"])
        gx = OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
        sess_run(sess, gx)
        obs = dict(
            shape=str(shape).replace(
                " ", ""), slice=str(slices).replace(
                " ", ""))
        r = measure_time(
            'sess_run(sess, gx)',
            number=number,
            div_by_number=True,
            context={
                'sess': sess,
                'gx': gx,
                'sess_run': sess_run})
        obs.update(r)
        obs['provider'] = 'GPU'
        data.append(obs)

df = DataFrame(data)
print(df)

########################################
# Better display
# ++++++++++++++

piv = pivot_table(
    df,
    index=[
        "shape",
        "slice"],
    columns="provider",
    values="average")
if 'GPU' in piv.columns:
    piv['ratio'] = piv['GPU'] / piv['CPU']
print(piv)

#############################
# Graphs.

piv.plot()
