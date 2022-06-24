"""
.. _l-benchmark-onnx-serialize:

SerializeToString and ParseFromString
=====================================


.. contents::
    :local:

Startup
+++++++

This section creates an ONNX graph if there is not one.

"""
import time
import numpy
import onnx
from pyquickhelper.pycode.profiling import profile, profile2graph
from cpyquickhelper.numbers.speed_measure import measure_time
import matplotlib.pyplot as plt
import pandas
from tqdm import tqdm
from onnx.checker import check_model
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.npy.xop import loadop
try:
    from mlprodict.onnx_tools._onnx_check_model import check_model as check_model_py
except ImportError:
    check_model_py = None


############################################
# Available optimisation on this machine.

print(code_optimisation())


##############################
# Build an ONNX graph of different size
# +++++++++++++++++++++++++++++++++++++

def build_model(n_nodes, size, opv=15):
    OnnxAdd, OnnxIdentity = loadop('Add', 'Identity')
    x = 'X'
    for n in range(n_nodes):
        y = OnnxAdd(x, numpy.random.randn(size).astype(numpy.float32),
                    op_version=opv)
        x = y
    final = OnnxIdentity(x, op_version=opv, output_names=['Y'])
    x = numpy.zeros((10, 10), dtype=numpy.float32)
    return final.to_onnx({'X': x}, {'Y': x}, target_opset=opv)


model = build_model(2, 5)
print(onnx_simple_text_plot(model))

##########################################
# Measure the time of serialization functions
# +++++++++++++++++++++++++++++++++++++++++++


def parse(buffer):
    proto = onnx.ModelProto()
    proto.ParseFromString(buffer)
    return proto


data = []
nodes = [5, 10, 20]
for size in tqdm([10, 100, 1000, 10000, 100000, 200000, 300000]):
    for n_nodes in nodes:
        repeat = 20 if size < 100000 else 5
        onx = build_model(n_nodes, size)
        serialized = onx.SerializeToString()
        onnx_size = len(serialized)
        obs = measure_time(lambda: onx.SerializeToString(),
                           div_by_number=True, repeat=repeat)
        obs['size'] = size
        obs['n_nodes'] = n_nodes
        obs['onnx_size'] = onnx_size
        obs['task'] = "SerializeToString"
        data.append(obs)

        parsed = parse(serialized)
        obs = measure_time(lambda: parse(serialized),
                           div_by_number=True, repeat=repeat)
        obs['size'] = size
        obs['n_nodes'] = n_nodes
        obs['onnx_size'] = onnx_size
        obs['task'] = "ParseFromString"
        data.append(obs)

        obs = measure_time(lambda: check_model(onx, full_check=False),
                           div_by_number=True, repeat=repeat)
        obs['size'] = size
        obs['n_nodes'] = n_nodes
        obs['onnx_size'] = onnx_size
        obs['task'] = "check_model"
        data.append(obs)

        if check_model_py is None:
            continue

        obs = measure_time(lambda: check_model_py(onx),
                           div_by_number=True, repeat=repeat)
        obs['size'] = size
        obs['n_nodes'] = n_nodes
        obs['onnx_size'] = onnx_size
        obs['task'] = "check_model_py"
        data.append(obs)

############################################
# time
df = pandas.DataFrame(data).sort_values(
    ['task', 'onnx_size', 'size', 'n_nodes'])
df[['task', 'onnx_size', 'size', 'n_nodes', 'average']]

###############################################
# Summary
# +++++++

df.to_excel("time.xlsx", index=False)
piv = df.pivot(index='onnx_size', columns='task', values='average')
piv

#########################################
# Graph
# +++++

fig, ax = plt.subplots(1, 1)
piv.plot(title="Time processing of serialization functions\n"
               "lower better", ax=ax)
ax.set_xlabel("onnx size")
ax.set_ylabel("s")

###########################################
# Conclusion
# ++++++++++
#
# This graph shows that implementing check_model in python is much slower
# than the C++ version. However, protobuf prevents from sharing
# ModelProto from Python to C++ (see `Python Updates
# <https://developers.google.com/protocol-buffers/docs/news/2022-05-06>`_)
# unless the python package is compiled with a specific setting
# (problably slower). A profiling shows that the code spends quite some time
# in function :func:`getattr`.

ps = profile(lambda: check_model_py(onx))[0]
root, nodes = profile2graph(ps, clean_text=lambda x: x.split('/')[-1])
text = root.to_text()
print(text)

plt.show()
