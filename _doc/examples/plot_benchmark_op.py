"""
.. _example-ort-training:

Benchmark and profile of operator Slice
=======================================

This short code compares the execution of the operator *Slice*
between :epkg:`numpy` and :epkg:`onnxruntime` for three
configurations.

.. contents::
    :local:

A simple example
++++++++++++++++

"""

import json
import numpy
from numpy.testing import assert_almost_equal
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession, get_device, SessionOptions
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxSlice, OnnxAdd, OnnxMul
from cpyquickhelper.numbers import measure_time
from tqdm import tqdm
from mlprodict.testing.experimental_c_impl.experimental_c import (
    code_optimisation)
from mlprodict.onnxrt.ops_whole.session import OnnxWholeSession
from onnxcustom.utils.onnxruntime_helper import get_ort_device

print([code_optimisation(), get_device()])


###################################
# The functions to compare.

def build_ort_op(op_version=14, save=None, **kwargs):  # opset=13, 14, ...
    slices = kwargs['slices']
    slice1, slice2 = slices
    slice1 = slice(0, None) if slice1 is None else slice(*slice1)
    slice2 = slice(0, None) if slice2 is None else slice(*slice2)

    axes = []
    starts = []
    ends = []
    for i in [0, 1]:
        if slices[i] is None:
            continue
        axes.append(i)
        starts.append(slices[i][0])
        ends.append(slices[i][1])
    starts = numpy.array(starts, dtype=numpy.int64)
    ends = numpy.array(ends, dtype=numpy.int64)
    axes = numpy.array(axes, dtype=numpy.int64)
    node1 = OnnxSlice('X', starts, ends, axes, op_version=op_version)
    node2 = OnnxAdd(node1, numpy.array([1], dtype=numpy.float32),
                    op_version=op_version)
    node3 = OnnxSlice(node2, starts, ends, axes,
                      op_version=op_version)
    node4 = OnnxMul(node3, numpy.array([2], dtype=numpy.float32),
                    op_version=op_version, output_names=['Y'])
    onx = node4.to_onnx(inputs=[('X', FloatTensorType([None, None]))],
                        target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString(),
                            providers=["CPUExecutionProvider"])
    if save is not None:
        with open(save, "wb") as f:
            f.write(onx.SerializeToString())

    def npy_fct(x):
        return ((x[slice1, slice2] + 1)[slice1, slice2] * 2).copy()

    rnd = numpy.random.randn(10, 10).astype(numpy.float32)
    expected = npy_fct(rnd)
    got = sess.run(None, {'X': rnd})[0]
    try:
        assert_almost_equal(expected, got)
    except AssertionError as e:
        raise AssertionError(
            "kwargs=%r slice1=%r slice2=%r shapes=%r ? %r "
            "(x[slice1, slice2].shape)=%r" % (
                kwargs, slice1, slice2, expected.shape,
                got.shape, rnd[slice1, slice2].shape)) from e

    if get_device().upper() == 'GPU':
        sessg = InferenceSession(onx.SerializeToString(),
                                 providers=["CUDAExecutionProvider"])
        io_binding = sessg.io_binding()._iobinding
        device = get_ort_device('cuda:0')

        def run_gpu(x):
            io_binding.bind_input(
                'X', device, numpy.float32, x.shape(), x.data_ptr())
            io_binding.bind_output('Y', device)
            return sessg._sess.run_with_iobinding(io_binding, None)

        return onx, lambda x: sess.run(None, {'X': x}), npy_fct, run_gpu
    else:
        return onx, lambda x: sess.run(None, {'X': x}), npy_fct, None


###########################################
# The benchmark.


def loop_fct(fct, xs):
    for x in xs:
        fct(x)


def benchmark_op(repeat=10, number=10, name="Slice", shape_slice_fct=None,
                 save=None, opset=14, repeat_profile=1500, verbose=1):
    if verbose:
        print("[benchmark_op] start repeat=%d number=%d repeat_profile=%d"
              " opset=%d." % (repeat, number, repeat_profile, opset))
    res = []
    for dim in tqdm([8, 16, 32, 64, 100, 128, 200,
                     256, 400, 512, 600, 784, 800,
                     1000, 1024, 1200]):
        shape, slices = shape_slice_fct(dim)
        onx, ort_fct, npy_fct, ort_fct_gpu = build_ort_op(
            save=save, op_version=opset, slices=slices)

        n_arrays = 20
        if dim >= 512:
            n_arrays = 10
        xs = [numpy.random.rand(*shape).astype(numpy.float32)
              for _ in range(n_arrays)]
        info = dict(shape=shape)

        ctx = dict(xs=xs, loop_fct=loop_fct)

        # numpy
        ctx['fct'] = npy_fct
        obs = measure_time(
            lambda: loop_fct(npy_fct, xs),
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'numpy'
        obs['shape'] = ",".join(map(str, shape))
        obs['slices'] = str(slices)
        obs.update(info)
        res.append(obs)

        # onnxruntime
        ctx['fct'] = ort_fct
        obs = measure_time(
            lambda: loop_fct(ort_fct, xs),
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'ort'
        obs['shape'] = ",".join(map(str, shape))
        obs['slices'] = str(slices)
        obs.update(info)
        res.append(obs)

        if ort_fct_gpu is not None:

            # onnxruntime
            dev = get_ort_device('cuda:0')
            ctx['xs'] = [
                C_OrtValue.ortvalue_from_numpy(x, dev)
                for x in xs]
            ctx['fct'] = ort_fct_gpu
            obs = measure_time(
                lambda: loop_fct(ort_fct_gpu, ctx['xs']),
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'ort_gpu'
            obs['shape'] = ",".join(map(str, shape))
            obs['slices'] = str(slices)
            obs.update(info)
            res.append(obs)

    # profiling CPU
    if verbose:
        print("[benchmark_op] done.")
        print("[benchmark_op] profile CPU.")
    so = SessionOptions()
    so.enable_profiling = True
    sess = InferenceSession(onx.SerializeToString(), so,
                            providers=["CPUExecutionProvider"])
    for i in range(0, repeat_profile):
        sess.run(None, {'X': xs[-1]}, )
    prof = sess.end_profiling()
    with open(prof, "r") as f:
        js = json.load(f)
    dfprof = DataFrame(OnnxWholeSession.process_profiling(js))
    dfprof['shape'] = ",".join(map(str, shape))
    dfprof['slices'] = str(slices)
    if verbose:
        print("[benchmark_op] done.")

    # profiling CPU
    if ort_fct_gpu is not None:
        if verbose:
            print("[benchmark_op] profile GPU.")
        so = SessionOptions()
        so.enable_profiling = True
        sess = InferenceSession(onx.SerializeToString(), so,
                                providers=["CUDAExecutionProvider"])
        io_binding = sess.io_binding()._iobinding
        device = get_ort_device('cpu')

        for i in range(0, repeat_profile):
            x = ctx['xs'][-1]
            io_binding.bind_input(
                'X', device, numpy.float32, x.shape(), x.data_ptr())
            io_binding.bind_output('Y', device)
            sess._sess.run_with_iobinding(io_binding, None)

        prof = sess.end_profiling()
        with open(prof, "r") as f:
            js = json.load(f)
        dfprofgpu = DataFrame(OnnxWholeSession.process_profiling(js))
        dfprofgpu['shape'] = ",".join(map(str, shape))
        dfprofgpu['slices'] = str(slices)
        if verbose:
            print("[benchmark_op] profile done.")
    else:
        dfprofgpu = None

    # Dataframes
    shape_name = str(shape).replace(str(dim), "N")
    df = pandas.DataFrame(res)
    piv = df.pivot('shape', 'fct', 'average')

    rs = piv.copy()
    for c in ['numpy', 'ort', 'ort_gpu']:
        if c in rs.columns:
            rs["numpy/%s" % c] = rs['numpy'] / rs[c]
    rs = rs[[c for c in rs.columns if "/numpy" not in c]].copy()

    # Graphs.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv.plot(logx=True, logy=True, ax=ax[0],
             title="%s benchmark\n%r"
                   " lower better" % (name, shape_name))
    ax[0].legend(prop={"size": 9})
    rs.plot(logx=True, logy=True, ax=ax[1],
            title="%s Speedup, baseline=numpy\n%r"
                  " higher better" % (name, shape_name))
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
    ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
    ax[1].legend(prop={"size": 9})
    return dfprof, dfprofgpu, df, rs, ax


######################################
# The results.

nth = int(code_optimisation().split('=')[1])
cols_profile = ["shape", "slices", "args_op_name", 'args_provider']

##############################################
# shape = (100, N) - slice = [1:-1], :
# ++++++++++++++++++++++++++++++++++++

dfs = []
dfprof, dfprofgpu, df, piv, ax = benchmark_op(
    shape_slice_fct=lambda dim: ((256, dim), ((1, -1), None)),
    save="bslice.onnx", number=nth * 4, repeat=8, repeat_profile=100 * nth)

dfs.append(df)
piv2 = df.pivot("fct", "shape", "average")
print("slices = [1:-1], :")
print(piv.to_markdown())
print(dfprof.drop(['pid', 'tid', 'ts'], axis=1).groupby(
    cols_profile).sum().to_markdown())
if dfprofgpu is not None:
    print(dfprofgpu.drop(['pid', 'tid'], axis=1).groupby(
        cols_profile).sum().to_markdown())

##############################################
# shape = (100, N) - slice = :, [1:-1]
# ++++++++++++++++++++++++++++++++++++

dfs = []
dfprof, dfprofgpu, df, piv, ax = benchmark_op(
    shape_slice_fct=lambda dim: ((256, dim), (None, (1, -1))),
    save="bslice.onnx", number=nth * 4, repeat=8, repeat_profile=100 * nth)

dfs.append(df)
piv2 = df.pivot("fct", "shape", "average")
print("slices = :, [1:-1]")
print(piv.to_markdown())
print(dfprof.drop(['pid', 'tid', 'ts'], axis=1).groupby(
    cols_profile).sum().to_markdown())
if dfprofgpu is not None:
    print(dfprofgpu.drop(['pid', 'tid'], axis=1).groupby(
        cols_profile).sum().to_markdown())

##############################################
# shape = (100, N) - slice = [1:-1], [1:-1]
# +++++++++++++++++++++++++++++++++++++++++

dfs = []
dfprof, dfprofgpu, df, piv, ax = benchmark_op(
    shape_slice_fct=lambda dim: ((256, dim), ((1, -1), (1, -1))),
    save="bslice.onnx", number=nth * 4, repeat=8, repeat_profile=100 * nth)

dfs.append(df)
piv2 = df.pivot("fct", "shape", "average")
print("slices = [1:-1], [1:-1]")
print(piv.to_markdown())
print(dfprof.drop(['pid', 'tid', 'ts'], axis=1).groupby(
    cols_profile).sum().to_markdown())
if dfprofgpu is not None:
    print(dfprofgpu.drop(['pid', 'tid'], axis=1).groupby(
        cols_profile).sum().to_markdown())
