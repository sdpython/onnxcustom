"""
.. _l-profile-ort-api:

Profile onnxruntime execution
=============================

The following examples converts a model into :epkg:`ONNX` and runs it
with :epkg:`onnxruntime`. This one is then uses to profile the execution
by looking the time spent in each operator. This analysis gives some
hints on how to optimize the processing time by looking the nodes
consuming most of the ressources.

.. contents::
    :local:

Neareast Neighbours
+++++++++++++++++++

"""
import json
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import pandas
from onnxruntime import InferenceSession, SessionOptions, get_device
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    SessionIOBinding, OrtDevice as C_OrtDevice, OrtValue as C_OrtValue)
from sklearn.neighbors import RadiusNeighborsRegressor
from skl2onnx import to_onnx
from tqdm import tqdm
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
from mlprodict.plotting.plotting import onnx_simple_text_plot, plot_onnx
from mlprodict.onnxrt.ops_whole.session import OnnxWholeSession

############################################
# Available optimisation on this machine.

print(code_optimisation())


##############################
# Building the model
# ++++++++++++++++++

X = numpy.random.randn(1000, 10).astype(numpy.float64)
y = X.sum(axis=1).reshape((-1, 1))

model = RadiusNeighborsRegressor()
model.fit(X, y)

#################################
# Conversion to ONNX
# ++++++++++++++++++

onx = to_onnx(model, X, options={'optim': 'cdist'})

print(onnx_simple_text_plot(onx))

##################################
# The ONNX graph looks like the following.

_, ax = plt.subplots(1, 1, figsize=(8, 15))
plot_onnx(onx, ax=ax)


###################################
# Profiling
# +++++++++
#
# The profiling is enabled by setting attribute `enable_profling`
# in :epkg:`SessionOptions`. Method *end_profiling* collects
# all the results and stores it on disk in :epkg:`JSON` format.

so = SessionOptions()
so.enable_profiling = True
sess = InferenceSession(onx.SerializeToString(), so,
                        providers=['CPUExecutionProvider'])
feeds = {'X': X[:100]}

for i in tqdm(range(0, 10)):
    sess.run(None, feeds)

prof = sess.end_profiling()
print(prof)

######################################
# Better rendering
# ++++++++++++++++


with open(prof, "r") as f:
    js = json.load(f)
df = pandas.DataFrame(OnnxWholeSession.process_profiling(js))
df

######################################
# Graphs
# ++++++
#
# First graph is by operator type.

gr_dur = df[['dur', "args_op_name"]].groupby(
    "args_op_name").sum().sort_values('dur')
gr_n = df[['dur', "args_op_name"]].groupby(
    "args_op_name").count().sort_values('dur')
gr_n = gr_n.loc[gr_dur.index, :]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
gr_dur.plot.barh(ax=ax[0])
gr_n.plot.barh(ax=ax[1])
ax[0].set_title("duration")
ax[1].set_title("n occurences")
fig.suptitle(model.__class__.__name__)

######################################
# Second graph is by operator name.

gr_dur = df[['dur', "args_op_name", "name"]].groupby(
    ["args_op_name", "name"]).sum().sort_values('dur')
gr_dur.head(n=5)

#######################################
# And the graph.

_, ax = plt.subplots(1, 1, figsize=(8, gr_dur.shape[0] // 2))
gr_dur.plot.barh(ax=ax)
ax.set_title("duration per node")
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(7)
make_axes_area_auto_adjustable(ax)

##########################################
# The model spends most of its time in CumSum operator.
# Operator Shape gets called the highest number of times.


# plt.show()

##########################################
# GPU or CPU
# ++++++++++

if get_device().upper() == 'GPU':
    ort_device = C_OrtDevice(
        C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)
else:
    ort_device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)

# session
sess = InferenceSession(onx.SerializeToString(), so,
                        providers=['CPUExecutionProvider',
                                   'CUDAExecutionProvider'])
bind = SessionIOBinding(sess._sess)

# moving the data on CPU or GPU
ort_value = C_OrtValue.ortvalue_from_numpy(X, ort_device)

#######################################
# A function which calls the API for any device.


def run_with_iobinding(sess, bind, ort_device, ort_value, dtype):
    bind.bind_input('X', ort_device, dtype, ort_value.shape(),
                    ort_value.data_ptr())
    bind.bind_output('variable', ort_device)
    sess._sess.run_with_iobinding(bind, None)
    ortvalues = bind.get_outputs()
    return ortvalues[0].numpy()

#######################################
# The profiling.


for i in tqdm(range(0, 10)):
    run_with_iobinding(sess, bind, ort_device, ort_value, X.dtype)

prof = sess.end_profiling()
with open(prof, "r") as f:
    js = json.load(f)
df = pandas.DataFrame(OnnxWholeSession.process_profiling(js))
df

###################################
# First graph is by operator type.

gr_dur = df[['dur', "args_op_name"]].groupby(
    "args_op_name").sum().sort_values('dur')
gr_n = df[['dur', "args_op_name"]].groupby(
    "args_op_name").count().sort_values('dur')
gr_n = gr_n.loc[gr_dur.index, :]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
gr_dur.plot.barh(ax=ax[0])
gr_n.plot.barh(ax=ax[1])
ax[0].set_title("duration")
ax[1].set_title("n occurences")
fig.suptitle(model.__class__.__name__)

######################################
# Second graph is by operator name.

gr_dur = df[['dur', "args_op_name", "name"]].groupby(
    ["args_op_name", "name"]).sum().sort_values('dur')
gr_dur.head(n=5)

#######################################
# And the graph.

_, ax = plt.subplots(1, 1, figsize=(8, gr_dur.shape[0] // 2))
gr_dur.plot.barh(ax=ax)
ax.set_title("duration per node")
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(7)
make_axes_area_auto_adjustable(ax)

##########################################
# It shows the same results.

# plt.show()
