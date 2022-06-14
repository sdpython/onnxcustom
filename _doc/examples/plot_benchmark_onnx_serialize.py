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
import numpy
import onnx
from cpyquickhelper.numbers.speed_measure import measure_time
import matplotlib.pyplot as plt
import pandas
from tqdm import tqdm
from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.npy.xop import loadop


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
    return final.to_onnx(numpy.float32, numpy.float32, target_opset=opv)


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
        onx = build_model(n_nodes, size)
        serialized = onx.SerializeToString()
        onnx_size = len(serialized)
        obs = measure_time(lambda: onx.SerializeToString(), div_by_number=True, repeat=20)
        obs['size'] = size
        obs['n_nodes'] = n_nodes
        obs['onnx_size'] = onnx_size
        obs['task'] = "SerializeToString"
        data.append(obs)

        parsed = parse(serialized)
        obs = measure_time(lambda: parse(serialized), div_by_number=True, repeat=20)
        obs['size'] = size
        obs['n_nodes'] = n_nodes
        obs['onnx_size'] = onnx_size
        obs['task'] = "ParseFromString"
        data.append(obs)


df = pandas.DataFrame(data).sort_values(['task', 'onnx_size', 'size', 'n_nodes'])
df[['task', 'onnx_size', 'size', 'n_nodes', 'average']]


###############################################
# Summary
# +++++++

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

plt.show()
