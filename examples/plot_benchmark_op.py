"""
.. _example-ort-training:

Benchmark an operator
=====================



.. contents::
    :local:

A simple example
++++++++++++++++

"""

import numpy
import pandas
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession, get_device
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxSlice, OnnxAdd, OnnxMul
from mlprodict.tools import measure_time
from tqdm import tqdm
from mlprodict.testing.experimental_c import code_optimisation
print(code_optimisation(), get_device())


###################################
# The functions to compare.

def build_ort_op(op_version=14, save=None):  # opset=13, 14, ...
    starts = numpy.array([1, -1], dtype=numpy.int64)
    ends = starts
    node1 = OnnxSlice('X', starts, ends, op_version=op_version)
    node2 = OnnxAdd(node1, numpy.array([1], dtype=numpy.float32),
                    op_version=op_version)
    node3 = OnnxSlice(node2, starts, ends,
                      op_version=op_version)
    node4 = OnnxMul(node3, numpy.array([2], dtype=numpy.float32),
                    op_version=op_version, output_names=['Y'])
    onx = node4.to_onnx(inputs=[('X', FloatTensorType([None, None]))],
                        target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    if save is not None:
        with open(save, "wb") as f:
            f.write(onx.SerializeToString())

    def npy_fct(x): return (x[1:-1, 1:-1] + 1)[1:-1, 1:-1] * 2
    return lambda x: sess.run(None, {'X': x}), npy_fct


###########################################
# The benchmark.


def loop_fct(fct, xs):
    for x in xs:
        fct(x)


def benchmark_op(repeat=2, number=5, name="Slice", shape_fct=None,
                 save=None, opset=14):
    ort_fct, npy_fct = build_ort_op(save=save, op_version=opset)
    res = []
    for dim in tqdm([8, 16, 32, 64, 100, 128, 200,
                     256, 400, 512, 1024]):
        shape = shape_fct(dim)
        n_arrays = 10 if dim < 512 else 4
        xs = [numpy.random.rand(*shape).astype(numpy.float32)
              for _ in range(n_arrays)]
        info = dict(shape=shape)

        ctx = dict(xs=xs, loop_fct=loop_fct)

        # numpy
        ctx['fct'] = npy_fct
        obs = measure_time(
            "loop_fct(fct, xs)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'numpy'
        obs.update(info)
        res.append(obs)

        # onnxruntime
        ctx['fct'] = ort_fct
        obs = measure_time(
            "loop_fct(fct, xs)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'ort'
        obs.update(info)
        res.append(obs)

    # Dataframes
    shape_name = str(shape).replace(str(dim), "N")
    df = pandas.DataFrame(res)
    df.columns = [_.replace('dim', 'N') for _ in df.columns]
    piv = df.pivot('N', 'fct', 'average')

    rs = piv.copy()
    for c in ['ort']:
        if c in rs.columns:
            rs[c] = rs['numpy'] / rs[c]
    rs['numpy'] = 1.

    # Graphs.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv.plot(logx=True, logy=True, ax=ax[0],
             title="%s benchmark\n%r - %r"
                   " lower better" % (name, shape_name, axes))
    ax[0].legend(prop={"size": 9})
    rs.plot(logx=True, logy=True, ax=ax[1],
            title="%s Speedup, baseline=numpy\n%r - %r"
                  " higher better" % (name, shape_name, axes))
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
    ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
    ax[1].legend(prop={"size": 9})
    return df, rs, ax


######################################
# The results.

axes = (3, )
dfs = []
df, piv, ax = benchmark_op(
    shape_fct=lambda dim: (
        dim, dim), save="bslice.onnx")
dfs.append(df)
piv2 = df.pivot("fct", "N", "average")
print(piv)
print(piv2.T)
