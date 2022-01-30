"""
.. _benchmark-inference-onnx-function:

Compares numpy to onnxruntime on simple functions
=================================================

:epkg:`onnxruntime` can be used a replacement to :epkg:`numpy`.
It can be used to implement a training algorithm,
:epkg:`onnxruntime-training` differentiate an onnx graph and
runs it to compute the gradient. Simple functions are implemented
in ONNX and ran with :epkg:`onnxruntime` to update the weights.
:func:`function_onnx_graph
<onnxcustom.utils.onnx_function.function_onnx_graph>` returns many
functions used to implement a training algorithm.
The following benchmarks compares a couple of implementations:

* `numpy`: an implementation based on numpy, not optimized
* `sess`: inference through an ONNX graph executed with
  method `onnxruntime.InferenceSession.run`
* `bind`: inference through an ONNX graph executed with
  method `onnxruntime.InferenceSession.run_with_iobinding`
* `run`: inference through an ONNX graph executed with
  method `onnxruntime.InferenceSession.run_with_iobinding`
  but without counting the binding assuming input buffers
  are reused and do not need binding again

.. contents::
    :local:

axpy
++++

This function implements :math:`Y = f(X1, X2, \\alpha) = \\alpha X1 + X2`.

"""
import numpy
from scipy.special import expit
import pandas
from tqdm import tqdm
from cpyquickhelper.numbers.speed_measure import measure_time
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    SessionIOBinding, OrtDevice as C_OrtDevice,
    OrtValue as C_OrtValue)
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from onnxcustom.utils.onnx_function import function_onnx_graph

fct_onx = function_onnx_graph("axpy")
print(onnx_simple_text_plot(fct_onx))

###########################################
# The numpy implementation is the following.

fct_numpy = lambda X1, X2, alpha: X1 * alpha + X2

###########################################
# The benchmark


def reshape(a, dim):
    if len(a.shape) == 2:
        return a[:dim].copy()
    return a


def bind_and_run(sess, bind, names, args, out_names, device):
    for n, a in zip(names, args):
        bind.bind_ortvalue_input(n, a)
    for o in out_names:
        bind.bind_output(o, device)
    sess.run_with_iobinding(bind, None)
    return bind.get_outputs()


def nobind_just_run(sess, bind):
    sess.run_with_iobinding(bind, None)


def benchmark(name, onx, fct_numpy, *args,
              dims=(1, 10, 100, 200, 500, 1000, 2000, 10000)):
    sess = InferenceSession(onx.SerializeToString())
    device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    if len(names) != len(args):
        raise RuntimeError(
            "Size mismatch %d != %d." % (len(names), len(args)))

    rows = []
    for dim in tqdm(dims):
        new_args = [reshape(a, dim) for a in args]
        ortvalues = [
            C_OrtValue.ortvalue_from_numpy(a, device)
            for a in new_args]

        ms = measure_time(lambda: fct_numpy(*new_args),
                          repeat=50, number=100)
        ms.update(dict(name=name, impl='numpy', dim=dim))
        rows.append(ms)

        inps = {n: a for n, a in zip(names, new_args)}
        ms = measure_time(lambda: sess.run(None, inps))
        ms.update(dict(name=name, impl='sess', dim=dim))
        rows.append(ms)

        bind = SessionIOBinding(sess._sess)
        ms = measure_time(
            lambda: bind_and_run(
                sess._sess, bind, names, ortvalues, out_names, device))
        ms.update(dict(name=name, impl='bind_run', dim=dim))
        rows.append(ms)

        ms = measure_time(
            lambda: nobind_just_run(sess._sess, bind))
        ms.update(dict(name=name, impl='run', dim=dim))
        rows.append(ms)

    return rows

################################
# Back to function axpy.


rows = benchmark(
    'axpy', fct_onx, fct_numpy,
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.array([0.5], dtype=numpy.float32))

all_rows = []
all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv

#####################################
# Graph.

name = rows[0]['name']
ax = piv.plot(logx=True, logy=True)
ax.set_title(name + "\nlower is better")


#########################################
# axpyw
# +++++
#
# It does :math:`Y, Z = f(X1, X2, G, \alpha, \beta) = (Y, Z)`
# where :math:`Z = \beta G + \alpha X1` and
# :math:`Y = Z + X2`.


fct_onx = function_onnx_graph("axpyw")
print(onnx_simple_text_plot(fct_onx))

#####################################
# benchmark

fct_numpy = lambda x1, x2, g, alpha, beta: (
    x1 * alpha + x2 + beta * g, x1 * alpha + beta * g)

rows = benchmark(
    'axpyw', fct_onx, fct_numpy,
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.array([0.5], dtype=numpy.float32),
    numpy.array([0.5], dtype=numpy.float32))

all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv

#####################################
# Graph.

name = rows[0]['name']
ax = piv.plot(logx=True, logy=True)
ax.set_title(name + "\nlower is better")


#######################################
# axpyw2
# ++++++
#
# It implements :math:`Y, Z = f(X1, X2, G, \alpha, \beta) = (Y, Z)`
# where :math:`Z = \beta G + \alpha X1` and
# :math:`Y = \beta * Z + \alpha X1 + X2`.

fct_onx = function_onnx_graph("axpyw2")
print(onnx_simple_text_plot(fct_onx))

#####################################
# benchmark

fct_numpy = lambda x1, x2, g, alpha, beta: (
    x1 * alpha + x2 + beta * (x1 * alpha + beta * g),
    x1 * alpha + beta * g)

rows = benchmark(
    'axpyw2', fct_onx, fct_numpy,
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.array([0.5], dtype=numpy.float32),
    numpy.array([0.5], dtype=numpy.float32))

all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv


#######################################
# copy
# ++++
#
# It implements a copy.

fct_onx = function_onnx_graph("copy")
print(onnx_simple_text_plot(fct_onx))

#####################################
# benchmark

fct_numpy = lambda x: x.copy()

rows = benchmark(
    'copy', fct_onx, fct_numpy,
    numpy.random.randn(1000, 10).astype(numpy.float32))

all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv

#####################################
# Graph.

name = rows[0]['name']
ax = piv.plot(logx=True, logy=True)
ax.set_title(name + "\nlower is better")


#######################################
# grad_loss_absolute_error
# ++++++++++++++++++++++++
#
# It implements :math:`Y = f(X1, X2) = \lVert X1 - X2 \rVert`.

fct_onx = function_onnx_graph("grad_loss_absolute_error")
print(onnx_simple_text_plot(fct_onx))

#####################################
# benchmark

fct_numpy = lambda x1, x2: (
    numpy.abs(x1 - x2).sum(), numpy.sign(x1 - x2))

rows = benchmark(
    'grad_loss_absolute_error', fct_onx, fct_numpy,
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.random.randn(1000, 10).astype(numpy.float32))

all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv

#####################################
# Graph.

name = rows[0]['name']
ax = piv.plot(logx=True, logy=True)
ax.set_title(name + "\nlower is better")


#######################################
# grad_loss_square_error
# ++++++++++++++++++++++
#
# It implements :math:`Y = f(X1, X2) = \lVert X1 - X2 \rVert^2`.

fct_onx = function_onnx_graph("grad_loss_square_error")
print(onnx_simple_text_plot(fct_onx))

#####################################
# benchmark

fct_numpy = lambda x1, x2: (
    ((x1 - x2) ** 2).sum(), (x1 - x2) * (-2))

rows = benchmark(
    'grad_loss_square_error', fct_onx, fct_numpy,
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.random.randn(1000, 10).astype(numpy.float32))

all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv

#####################################
# Graph.

name = rows[0]['name']
ax = piv.plot(logx=True, logy=True)
ax.set_title(name + "\nlower is better")


#######################################
# grad_loss_elastic_error
# +++++++++++++++++++++++
#
# It implements :math:`Y = f(X1, X2) = \beta \lVert X1 - X2 \rVert +
# \alpha \lVert X1 - X2 \rVert^2` or
# :math:`Y = f(X1, X2) = \beta \lVert w(X1 - X2) \rVert +
# \alpha \lVert (\sqrt{w}(X1 - X2) \rVert^2` if
# *weight_name* is not None and its gradient.
# *l1_weight* is :math:`\beta` and
# *l2_weight* is :math:`\alpha`.

fct_onx = function_onnx_graph("grad_loss_elastic_error")
print(onnx_simple_text_plot(fct_onx))

#####################################
# benchmark

fct_numpy = lambda x1, x2: (
    numpy.abs(x1 - x2).sum() * 0.1 + ((x1 - x2) ** 2).sum() * 0.9,
    numpy.sign(x1 - x2) * 0.1 - 2 * 0.9 * (x1 - x2))

rows = benchmark(
    'grad_loss_elastic_error', fct_onx, fct_numpy,
    numpy.random.randn(1000, 10).astype(numpy.float32),
    numpy.random.randn(1000, 10).astype(numpy.float32))

all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv


#####################################
# Graph.

name = rows[0]['name']
ax = piv.plot(logx=True, logy=True)
ax.set_title(name + "\nlower is better")


#######################################
# n_penalty_elastic_error
# +++++++++++++++++++++++
#
# It implements :math:`Y = f(W) = \beta \lVert W \rVert +
# \alpha \lVert W \rVert^2`
# *l1_weight* is :math:`\beta` and
# *l2_weight* is :math:`\alpha`.
# It does that for *n_tensors* and adds all of the results
# to an input loss.

fct_onx = function_onnx_graph("n_penalty_elastic_error")
print(onnx_simple_text_plot(fct_onx))

#####################################
# benchmark

fct_numpy = lambda loss, x: numpy.abs(x).sum() * 0.1 + ((x) ** 2).sum() * 0.9

rows = benchmark(
    'n_penalty_elastic_error', fct_onx, fct_numpy,
    numpy.array([[0.5]], dtype=numpy.float32),
    numpy.random.randn(1000, 10).astype(numpy.float32))

all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv

#####################################
# Graph.

name = rows[0]['name']
ax = piv.plot(logx=True, logy=True)
ax.set_title(name + "\nlower is better")


#######################################
# update_penalty_elastic_error
# ++++++++++++++++++++++++++++
#
# It implements :math:`Y = f(W) = W - 2 \beta W - \alpha sign(W)`
# *l1* is :math:`\beta` and
# *l2* is :math:`\alpha`.

fct_onx = function_onnx_graph("update_penalty_elastic_error")
print(onnx_simple_text_plot(fct_onx))

#####################################
# benchmark

fct_numpy = lambda x: numpy.sign(x) * 0.1 + (x * 0.9 * 2)

rows = benchmark(
    'update_penalty_elastic_error', fct_onx, fct_numpy,
    numpy.random.randn(1000, 10).astype(numpy.float32))

all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv


#####################################
# Graph.

name = rows[0]['name']
ax = piv.plot(logx=True, logy=True)
ax.set_title(name + "\nlower is better")


#######################################
# grad_sigmoid_neg_log_loss_error
# +++++++++++++++++++++++++++++++
#
# See :func:`_onnx_grad_sigmoid_neg_log_loss_error
# <onnxcustom.utils.onnx_function._onnx_grad_sigmoid_neg_log_loss_error>`.

fct_onx = function_onnx_graph("grad_sigmoid_neg_log_loss_error")
print(onnx_simple_text_plot(fct_onx))

#####################################
# benchmark


def loss(x1, x2, eps=1e-5):
    pr = expit(x2)
    cl = numpy.clip(pr, eps, 1 - eps)
    lo = - (1 - x1) * numpy.log(1 - cl) - x1 * numpy.log(cl)
    return lo


fct_numpy = lambda x1, x2: (loss(x1, x2).mean(), expit(x2) - x1)

rows = benchmark(
    'grad_sigmoid_neg_log_loss_error', fct_onx, fct_numpy,
    (numpy.random.randn(1000, 1) > 0).astype(numpy.int64),
    numpy.random.randn(1000, 10).astype(numpy.float32))

all_rows.extend(rows)
piv = pandas.DataFrame(rows).pivot('dim', 'impl', 'average')
piv

#####################################
# Graph.

name = rows[0]['name']
ax = piv.plot(logx=True, logy=True)
ax.set_title(name + "\nlower is better")


########################################
# Results
# +++++++

df = pandas.DataFrame(all_rows)
df

########################################
# Pivot

piv = pandas.pivot_table(
    df, index=['name', 'impl'], columns='dim', values='average')
piv
print(piv)

########################################
# Graph.

fig, ax = None, None


for i, name in enumerate(sorted(set(df['name']))):
    if fig is None:
        fig, ax = plt.subplots(2, 2, figsize=(8, 12), sharex=True)
    x, y = (i % 4) // 2, (i % 4) % 2
    piv = df[df.name == name].pivot('dim', 'impl', 'average')
    piv.plot(ax=ax[x, y], logx=True, logy=True)
    ax[x, y].set_title(name)
    ax[x, y].xaxis.set_label_text("")
    if i % 4 == 3:
        fig.suptitle("lower is better")
        fig.tight_layout()
        fig, ax = None, None


if fig is not None:
    fig.suptitle("lower is better")
    fig.tight_layout()


# plt.show()
