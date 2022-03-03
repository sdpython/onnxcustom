"""
.. _benchmark-ort-onnx-graph-opt:

Benchmark onnxruntime optimization
==================================

:epkg:`onnxruntime` does optimize the ONNX graph before
running the inference. It tries for example to fuse a matrix multiplication
following or followed by a transpose, choosing the most efficient path.

.. contents::
    :local:

One ONNX file
+++++++++++++

This section creates an ONNX graph if there is not one.

"""
import os
from collections import OrderedDict, Counter
import numpy
import onnx
from cpyquickhelper.numbers.speed_measure import measure_time
import pandas
from onnxruntime import InferenceSession, SessionOptions, get_device
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    SessionIOBinding, OrtDevice as C_OrtDevice, OrtValue as C_OrtValue,
    GraphOptimizationLevel)
from sklearn.neighbors import RadiusNeighborsRegressor
from skl2onnx import to_onnx
from tqdm import tqdm
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation


############################################
# Available optimisation on this machine.

print(code_optimisation())


##############################
# Building the model
# ++++++++++++++++++

filename = "onnx_to_profile.onnx"

if not os.path.exists(filename):
    print("Generate a graph for %r." % filename)
    X = numpy.random.randn(1000, 10).astype(numpy.float64)
    y = X.sum(axis=1).reshape((-1, 1))

    model = RadiusNeighborsRegressor()
    model.fit(X, y)
    onx = to_onnx(model, X, options={'optim': 'cdist'})

    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())

#####################################
# Functions
# +++++++++
#
# We need to generate random inputs to test the graph.


def random_input(typ, shape, batch):
    if typ == 'tensor(double)':
        dtype = numpy.float64
    elif typ == 'tensor(float)':
        dtype = numpy.float32
    else:
        raise NotImplementedError(
            "Unable to guess dtype from %r." % typ)

    if len(shape) <= 1:
        new_shape = shape
    elif shape[0] is None:
        new_shape = tuple([batch] + list(shape[1:]))
    else:
        new_shape = shape
    return numpy.random.randn(*new_shape).astype(dtype)


def random_feed(sess, batch=10):
    """
    Creates a dictionary of random inputs.

    :param batch: dimension to use as batch dimension if unknown
    :return: dictionary
    """
    inputs = sess.get_inputs()
    res = OrderedDict()
    for inp in inputs:
        name = inp.name
        typ = inp.type
        shape = inp.shape
        res[name] = random_input(typ, shape, batch)
    return res


#######################################
# A function which calls the API for any device.


def run_with_iobinding(sess, bind, ort_device, feed_ort_value, outputs):
    for name, (value, dtype) in feed_ort_value.items():
        bind.bind_input(name, ort_device, dtype, value.shape(),
                        value.data_ptr())
    for out in outputs:
        bind.bind_output(out, ort_device)
    sess._sess.run_with_iobinding(bind, None)
    ortvalues = bind.get_outputs()
    return [o.numpy() for o in ortvalues]


#################################
# Benchmark
# +++++++++
#
# Let's choose the device available on this machine.
# batch dimension is set to 10.

batch = 200

if get_device().upper() == 'GPU':
    ort_device = C_OrtDevice(
        C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)
    provider = 'CUDAExecutionProvider'
else:
    ort_device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    provider = 'CPUExecutionProvider'
print("provider = %r" % provider)

####################################
# We load the graph.

with open(filename, 'rb') as f:
    onx = onnx.load(f)

###############################
# Create of the session.
data = []
files = []
legend = []

for graph_opt, name_opt in tqdm([
        (GraphOptimizationLevel.ORT_DISABLE_ALL, "ORT_DISABLE_ALL"),
        (GraphOptimizationLevel.ORT_ENABLE_BASIC, "ORT_ENABLE_BASIC"),
        (GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "ORT_ENABLE_EXTENDED"),
        (GraphOptimizationLevel.ORT_ENABLE_ALL, "ORT_ENABLE_ALL")]):

    so = SessionOptions()
    so.graph_optimization_level = graph_opt
    so.optimized_model_filepath = (
        os.path.split(filename)[-1] + ".optimized.%s.onnx" % name_opt)
    files.append(so.optimized_model_filepath)
    legend.append(name_opt)
    sess = InferenceSession(onx.SerializeToString(), so,
                            providers=[provider])
    bind = SessionIOBinding(sess._sess)

    #####################################
    # Creates random data
    feed = random_feed(sess, batch)

    #####################################
    # moving the data on CPU or GPU
    feed_ort_value = OrderedDict(
        (name, (C_OrtValue.ortvalue_from_numpy(v, ort_device), v.dtype))
        for name, v in feed.items())
    outputs = [o.name for o in sess.get_outputs()]

    #######################################
    # The profiling.

    obs = measure_time(
        lambda: run_with_iobinding(
            sess, bind, ort_device, feed_ort_value, outputs),
        context=dict(run_with_iobinding=run_with_iobinding,
                     feed_ort_value=feed_ort_value, outputs=outputs,
                     sess=sess, bind=bind, ort_device=ort_device),
        repeat=10, number=10, div_by_number=True)
    obs['name'] = name_opt
    data.append(obs)


df = pandas.DataFrame(data)
df


##########################################
# Graph
# +++++

df = df.set_index('name')
dev = df[['deviation']].copy()
dev.columns = ['average']
ax = df[['average']].plot.bar(yerr=dev)
ax.set_title(os.path.split(filename)[-1])
ax.tick_params(axis='x', labelrotation=15)

###############################################
# The result are similar because the optimized model was very similar.

data = []
for name in files:
    with open(name, "rb") as f:
        onx = onnx.load(f)
    op_names = [op.op_type for op in onx.graph.node]
    data.append(Counter(op_names))

df = pandas.DataFrame(data).T
df.columns = legend
df

#########################################
# Graph.

ax = df.plot.barh(yerr=dev)
ax.set_title(os.path.split(filename)[-1])

# import matplotlib.pyplot as plt
# plt.show()
