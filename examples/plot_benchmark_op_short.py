"""
.. _example-ort-training:

Benchmark an operator
=====================



.. contents::
    :local:

A simple example
++++++++++++++++

"""

import json
import numpy
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession, get_device, OrtValue, SessionOptions
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxSlice, OnnxAdd, OnnxMul
from mlprodict.tools import measure_time
from tqdm import tqdm
from mlprodict.testing.experimental_c import code_optimisation
from mlprodict.onnxrt.ops_whole.session import OnnxWholeSession

print([code_optimisation(), get_device()])


###################################
# The functions to compare.

def build_ort_op(op_version=14, save=None):  # opset=13, 14, ...
    starts = numpy.array([1, 1], dtype=numpy.int64)
    ends = numpy.array([-1, -1], dtype=numpy.int64)
    node1 = OnnxSlice('X', starts, ends, op_version=op_version)
    node2 = OnnxAdd(node1, numpy.array([1], dtype=numpy.float32),
                    op_version=op_version)
    node3 = OnnxSlice(node2, starts, ends,
                      op_version=op_version)
    node4 = OnnxMul(node3, numpy.array([2], dtype=numpy.float32),
                    op_version=op_version, output_names=['Y'])
    onx = node4.to_onnx(inputs=[('X', FloatTensorType([None, None]))],
                        target_opset=op_version)
    return onx
    
    
onx = build_ort_op()

x = numpy.random.rand(50, 50).astype(numpy.float32)

from mlprodict.onnxrt import OnnxInference
oinf = OnnxInference(onx)
oinf.run({'X': x}, verbose=1, fLOG=print)

sess = InferenceSession(onx.SerializeToString(),
                        providers=["CPUExecutionProvider"])
print(sess.run(None, {'X': x})[0].shape)


sessg = InferenceSession(onx.SerializeToString(),
                         providers=["CUDAExecutionProvider"])

gx = OrtValue.ortvalue_from_numpy(x, 'cuda', 0)

io_binding = sessg.io_binding()
io_binding.bind_input(
    name='X', device_type=gx.device_name(), device_id=0,
    element_type=numpy.float32, shape=gx.shape(),
    buffer_ptr=gx.data_ptr())
io_binding.bind_output('Y')
sessg.run_with_iobinding(io_binding)
Y = io_binding.copy_outputs_to_cpu()[0]
print(Y.shape)


