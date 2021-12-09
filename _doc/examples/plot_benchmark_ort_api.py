"""
.. _benchmark-ort-api:

Benchmark onnxruntime API
=========================

This short code compares different ways to call onnxruntime API.

.. contents::
    :local:

Linear Regression
+++++++++++++++++

"""
import numpy
import pandas
from onnxruntime import InferenceSession
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    SessionIOBinding, OrtDevice as C_OrtDevice)
from sklearn import config_context
from sklearn.linear_model import LinearRegression
from skl2onnx import to_onnx
from cpyquickhelper.numbers.speed_measure import measure_time
from mlprodict.onnxrt import OnnxInference
from mlprodict.testing.experimental_c import code_optimisation

############################################
# Available optimisation on this machine.

print(code_optimisation())


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


#################################
# Benchmarks
# ++++++++++

data = []

###################################
# scikit-learn

with config_context(assume_finite=True):
    obs = measure_time(lambda: model.predict(X),
                       context=dict(model=model, X=X),
                       repeat=40, number=200)
    obs['name'] = 'skl'
    data.append(obs)


###################################
# numpy runtime
oinf = OnnxInference(onx, runtime="python_compiled")
obs = measure_time(lambda: oinf.run({'X': X}), context=dict(oinf=oinf, X=X),
                   repeat=40, number=200)
obs['name'] = 'numpy'
data.append(obs)


###################################
# onnxruntime: run
sess = InferenceSession(onx.SerializeToString())
obs = measure_time(lambda: sess.run(None, {'X': X}),
                   context=dict(sess=sess, X=X),
                   repeat=40, number=200)
obs['name'] = 'ort-run'
data.append(obs)


###################################
# onnxruntime: run_with_iobinding
sess = InferenceSession(onx.SerializeToString())
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
                   repeat=40, number=200)

obs['name'] = 'ort-bind'
data.append(obs)


###################################
# Final
# +++++

df = pandas.DataFrame(data)
print(df[['name', 'average', 'number', 'repeat']])
df

###################################
# Graph
# +++++

ax = df.set_index('name')[['average']].plot.bar()
ax.set_title("Average inference time")
ax.tick_params(axis='x', labelrotation=15)

###################################
# Conclusion
# ++++++++++
#
# A profiling (:epkg:`onnxruntime` is compiled with debug information)
# including # calls to native C++ functions shows that referencing input
# by name # takes a significant time when the graph is very small such
# as this one. # The logic in method *run_with_iobinding* is much longer
# that the one implemented in *run*.

# import matplotlib.pyplot as plt
# plt.show()
