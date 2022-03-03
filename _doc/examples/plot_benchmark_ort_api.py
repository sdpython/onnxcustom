"""
.. _benchmark-ort-api:

Benchmark onnxruntime API: run or ...
=====================================

This short code compares different methods to call onnxruntime API.

* `run`
* `run_with_ort_values`
* `run_with_iobinding`

You may profile this code:

::

    py-spy record -o plot_benchmark_ort_api.svg -r 10
    --native -- python plot_benchmark_ort_api.py


.. contents::
    :local:

Linear Regression
+++++++++++++++++

"""
import numpy
import pandas
from onnxruntime import InferenceSession
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    SessionIOBinding, OrtDevice as C_OrtDevice,
    OrtMemType, OrtValue as C_OrtValue, RunOptions)
from sklearn import config_context
from sklearn.linear_model import LinearRegression
from skl2onnx import to_onnx
from cpyquickhelper.numbers.speed_measure import measure_time
from mlprodict.onnxrt import OnnxInference
from mlprodict.plotting.plotting import onnx_simple_text_plot
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation

############################################
# Available optimisation on this machine.

print(code_optimisation())
repeat = 250
number = 250


##############################
# Building the model
# ++++++++++++++++++

X = numpy.random.randn(1000, 10).astype(numpy.float32)
y = X.sum(axis=1)

model = LinearRegression()
model.fit(X, y)

#################################
# Conversion to ONNX
# ++++++++++++++++++
onx = to_onnx(model, X, black_op={'LinearRegressor'})
print(onnx_simple_text_plot(onx))


#################################
# Benchmarks
# ++++++++++

data = []

###################################
# scikit-learn
print('scikit-learn')

with config_context(assume_finite=True):
    obs = measure_time(lambda: model.predict(X),
                       context=dict(model=model, X=X),
                       repeat=repeat, number=number)
    obs['name'] = 'skl'
    data.append(obs)


###################################
# numpy runtime
print('numpy')
oinf = OnnxInference(onx, runtime="python_compiled")
obs = measure_time(
    lambda: oinf.run({'X': X}), context=dict(oinf=oinf, X=X),
    repeat=repeat, number=number)
obs['name'] = 'numpy'
data.append(obs)


###################################
# onnxruntime: run
print('ort')
sess = InferenceSession(onx.SerializeToString(),
                        providers=['CPUExecutionProvider'])
obs = measure_time(lambda: sess.run(None, {'X': X}),
                   context=dict(sess=sess, X=X),
                   repeat=repeat, number=number)
obs['name'] = 'ort-run'
data.append(obs)


###################################
# onnxruntime: run
print('ort-c')
sess = InferenceSession(onx.SerializeToString(),
                        providers=['CPUExecutionProvider'])
ro = RunOptions()
output_names = [o.name for o in sess.get_outputs()]
obs = measure_time(
    lambda: sess._sess.run(output_names, {'X': X}, ro),
    context=dict(sess=sess, X=X),
    repeat=repeat, number=number)
obs['name'] = 'ort-c'
data.append(obs)


###################################
# onnxruntime: run_with_ort_values
print('ort-ov-c')
device = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)

Xov = C_OrtValue.ortvalue_from_numpy(X, device)

sess = InferenceSession(onx.SerializeToString(),
                        providers=['CPUExecutionProvider'])
ro = RunOptions()
output_names = [o.name for o in sess.get_outputs()]
obs = measure_time(
    lambda: sess._sess.run_with_ort_values(
        {'X': Xov}, output_names, ro),
    context=dict(sess=sess),
    repeat=repeat, number=number)
obs['name'] = 'ort-ov'
data.append(obs)


###################################
# onnxruntime: run_with_iobinding
print('ort-bind')
sess = InferenceSession(onx.SerializeToString(),
                        providers=['CPUExecutionProvider'])
bind = SessionIOBinding(sess._sess)
ort_device = C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)


def run_with_iobinding(sess, X, bind, ort_device):
    if X.__array_interface__['strides'] is not None:
        raise RuntimeError("onnxruntime only supports contiguous arrays.")
    bind.bind_input('X', ort_device, X.dtype, X.shape,
                    X.__array_interface__['data'][0])
    bind.bind_output('variable', ort_device)
    sess._sess.run_with_iobinding(bind, None)
    ortvalues = bind.get_outputs()
    return ortvalues[0].numpy()


obs = measure_time(lambda: run_with_iobinding(sess, X, bind, ort_device),
                   context=dict(run_with_iobinding=run_with_iobinding, X=X,
                                sess=sess, bind=bind, ort_device=ort_device),
                   repeat=repeat, number=number)

obs['name'] = 'ort-bind'
data.append(obs)


###################################
# This fourth implementation is very similar to the previous
# one but it only binds array once and reuse the memory
# without changing the binding. It assumes that input size
# and output size never change. It copies the data into
# the fixed buffer and returns the same array, modified
# inplace.

print('ort-bind-inplace')
sess = InferenceSession(onx.SerializeToString(),
                        providers=['CPUExecutionProvider'])
bind = SessionIOBinding(sess._sess)
ort_device = C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)

Y = sess.run(None, {'X': X})[0]
bX = X.copy()
bY = Y.copy()

bind.bind_input('X', ort_device, numpy.float32, bX.shape,
                bX.__array_interface__['data'][0])
bind.bind_output('variable', ort_device, numpy.float32, bY.shape,
                 bY.__array_interface__['data'][0])
ortvalues = bind.get_outputs()


def run_with_iobinding(sess, bX, bY, X, bind, ortvalues):
    if X.__array_interface__['strides'] is not None:
        raise RuntimeError("onnxruntime only supports contiguous arrays.")
    bX[:, :] = X[:, :]
    sess._sess.run_with_iobinding(bind, None)
    return bY


obs = measure_time(
    lambda: run_with_iobinding(
        sess, bX, bY, X, bind, ortvalues),
    context=dict(run_with_iobinding=run_with_iobinding, X=X,
                 sess=sess, bind=bind, ortvalues=ortvalues, bX=bX, bY=bY),
    repeat=repeat, number=number)

obs['name'] = 'ort-bind-inplace'
data.append(obs)


###################################
# Fifth implementation is equivalent to the previous one
# but does not copy anything.

print('ort-run-inplace')
sess = InferenceSession(onx.SerializeToString(),
                        providers=['CPUExecutionProvider'])
bind = SessionIOBinding(sess._sess)
ort_device = C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)

Y = sess.run(None, {'X': X})[0]
bX = X.copy()
bY = Y.copy()

bind.bind_input('X', ort_device, numpy.float32, bX.shape,
                bX.__array_interface__['data'][0])
bind.bind_output('variable', ort_device, numpy.float32, bY.shape,
                 bY.__array_interface__['data'][0])
ortvalues = bind.get_outputs()


def run_with_iobinding_no_copy(sess, bX, bY, X, bind, ortvalues):
    if X.__array_interface__['strides'] is not None:
        raise RuntimeError("onnxruntime only supports contiguous arrays.")
    # bX[:, :] = X[:, :]
    sess._sess.run_with_iobinding(bind, None)
    return bY


obs = measure_time(
    lambda: run_with_iobinding_no_copy(
        sess, bX, bY, X, bind, ortvalues),
    context=dict(run_with_iobinding_no_copy=run_with_iobinding_no_copy, X=X,
                 sess=sess, bind=bind, ortvalues=ortvalues, bX=bX, bY=bY),
    repeat=repeat, number=number)

obs['name'] = 'ort-run-inplace'
data.append(obs)


###################################
# Final
# +++++

df = pandas.DataFrame(data)
print(df[['name', 'average', 'number', 'repeat', 'deviation']])
df

###################################
# Graph
# +++++

ax = df.set_index('name')[['average']].plot.bar()
ax.set_title("Average inference time\nThe lower the better")
ax.tick_params(axis='x', labelrotation=15)

###################################
# Conclusion
# ++++++++++
#
# A profiling (:epkg:`onnxruntime` is compiled with debug information)
# including # calls to native C++ functions shows that referencing input
# by name # takes a significant time when the graph is very small such
# as this one. The logic in method *run_with_iobinding* is much longer
# that the one implemented in *run*.

# import matplotlib.pyplot as plt
# plt.show()
