
====================================
Inference with onnxruntime in Python
====================================

.. contents::
    :local:

Simple case
===========

The main class is :epkg:`InferenceSession`. It loads
an ONNX graph executes all the nodes in it.

.. runpython::
    :showcode:

    import numpy
    from onnxruntime import InferenceSession
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from skl2onnx import to_onnx

    # creation of an ONNX graph
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
    clr = LinearRegression()
    clr.fit(X_train, y_train)
    model_def = to_onnx(clr, X_train)

    # InferenceSession only accepts a file name or the serialized
    # ONNX graph.
    sess = InferenceSession(model_def.SerializeToString())

    # Method run takes two inputs, first one is
    # the list of desired outputs or None for all,
    # second is the input tensors in a dictionary
    result = sess.run(None, {'X': X_test[:5]})
    print(result)

    with open("linreg_model.onnx", "wb") as f:
        f.write(model_def.SerializeToString())

And visually:

.. gdot::
    :script: DOT-SECTION

    import onnx
    from mlprodict.onnxrt import OnnxInference

    with open("linreg_model.onnx", "rb") as f:
        onnx_model = onnx.load(f)
    print("DOT-SECTION", OnnxInference(onnx_model).to_dot(recursive=True))

Some informations about the graph can be retrieved
through the class :epkg:`InferenceSession` such as
inputs and outputs.

.. runpython::
    :showcode:

    from onnxruntime import InferenceSession

    sess = InferenceSession("linreg_model.onnx")

    for t in sess.get_inputs():
        print("input:", t.name, t.type, t.shape)

    for t in sess.get_outputs():
        print("output:", t.name, t.type, t.shape)

The class :epkg:`InferenceSession` is not pickable.

Inference on a device different from CPU
========================================

By default, everything happens on CPU.
Next lines shows how to do computation on GPU
with :epkg:`onnxruntime`.

C_OrtValue
++++++++++

IOBinding
+++++++++

Session Options
===============

Many options can change the behaviour of the class during predictions.
First class is :epkg:`SessionOptions`. It may change the default
behaviour. Next sections describe some of the members.
This class can also be used to profile the execution or
adjust graph optimization. This will be seen in further sections.
Next sections just give an overview, you should go to classes
:epkg:`SessionOptions` and :epkg:`RunOptions` to get the full list.

::

    from onnxruntime import InferenceSession, SessionOptions
    so = SessionOptions()
    # so.... =
    sess = InferenceSession(...., so)

logging
~~~~~~~

Parameters *log_severity_level* and *log_verbosity_level* may change
the verbosity level when the model is loaded.

The logging during execution can be modified with the same
attributes but in class :epkg:`RunOptions`. This class is given
to method `run`.

memory
~~~~~~

:epkg:`onnxruntime` focuses on efficiency first and memory peaks.
Following what should be the priority, following members
may be changed to trade efficiency against memory usage.

* *enable_cpu_mem_arena*: Enables the memory arena on CPU.
  Arena may pre-allocate memory for future usage.
  Set this option to false if you don’t want it.
  Default is True.

* *enable_mem_pattern*: Enable the memory pattern optimization.
  Default is true.

* *enable_mem_reuse*: Enable the memory reuse optimization.
  Default is true.

multithreading
~~~~~~~~~~~~~~

By default, :epkg:`onnxruntime` parallelizes the execution
within every node but does not run multiple node at the same time.
But that can be changed.

* *inter_op_num_threads*: Sets the number of threads used to
  parallelize the execution of the graph (across nodes).
  Default is 0 to let onnxruntime choose.

* *intra_op_num_threads*:  Sets the number of threads used to
  parallelize the execution within nodes.
  Default is 0 to let onnxruntime choose.

extensions
~~~~~~~~~~

Attribute `register_custom_ops_library` to register an
assembly implementing the runtime for custom nodes.
:epkg:`onnxruntime-extensions` is one of these extensions
mostly focusing on text processing (tokenizers) or simple
text manipulations.

Providers
=========

A provider is usually a list of implementation of ONNX operator
for a specific environment. `CPUExecutionProvider` provides implementations
for all operator on CPU. `CUDAExecutionProvider` does the same for GPU and
the CUDA drivers. The list of all providers depends on the compilation
settings. The list of available providers is a subset which depends on the machine
:epkg:`onnxruntime` is running on.

.. runpython::
    :showcode:

    import pprint
    import onnxruntime
    print("all providers")
    pprint.pprint(onnxruntime.get_all_providers())
    print("available providers")
    pprint.pprint(onnxruntime.get_available_providers())

:epkg:`onnxruntime` selects `CPUExecutionProvider` if its the only one available.
It raises an exception if there are more.
It is possible to select which providers must be used for the execution
by filling argument `providers`:

::

    sess = InferenceSession(
        ...
        providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
        ...)

All operators are not available in all providers, using multiple may improve
the processing time. Switching from one provider to another may mean
moving data from one memory manager to another, like the transition from CPU
to CUDA or the other way.

Profiling
=========

:epkg:`onnxruntime` offers the possibility to profile
the execution of a graph. It measures the time spent
in each operator. The user starts the profiling when
creating an instance of :epkg:`InferenceSession` and stops
it with method `end_profiling`. It stores the results
as a json file whose name is returned by the method.
The end of the example uses a tool to convert the json
into a table.

.. plot::
    :include-source:

    import json
    import numpy
    from pandas import DataFrame
    from onnxruntime import InferenceSession, RunOptions, SessionOptions
    from sklearn.datasets import make_classification
    from sklearn.cluster import KMeans
    from skl2onnx import to_onnx
    from mlprodict.onnxrt.ops_whole.session import OnnxWholeSession
    import matplotlib.pyplot as plt

    # creation of an ONNX graph.
    X, y = make_classification(100000)
    km = KMeans(max_iter=10)
    km.fit(X)
    onx = to_onnx(km, X[:1].astype(numpy.float32))

    # creation of a session that enables the profiling
    so = SessionOptions()
    so.enable_profiling = True
    sess = InferenceSession(onx.SerializeToString(), so)

    # execution
    for i in range(0, 111):
        sess.run(None, {'X': X.astype(numpy.float32)}, )

    # profiling ends
    prof = sess.end_profiling()
    # and is collected in that file:
    print(prof)

    # what does it look like?
    with open(prof, "r") as f:
        js = json.load(f)
    print(js[:3])

    # a tool to convert it into a table
    df = DataFrame(OnnxWholeSession.process_profiling(js))

    # it has the following columns
    print(df.columns)

    # and looks this way
    print(df.head(n=10))

    # but a graph is usually better...
    gr_dur = df[['dur', "args_op_name"]].groupby("args_op_name").sum().sort_values('dur')
    gr_n = df[['dur', "args_op_name"]].groupby("args_op_name").count().sort_values('dur')
    gr_n = gr_n.loc[gr_dur.index, :]

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    gr_dur.plot.barh(ax=ax[0])
    gr_dur /= gr_dur['dur'].sum()
    gr_dur.plot.barh(ax=ax[1])
    gr_n.plot.barh(ax=ax[2])
    ax[0].set_title("duration")
    ax[1].set_title("proportion")
    ax[2].set_title("n occurences");
    for a in ax:
        a.legend().set_visible(False)
        a.axis["left"].label.set_visible(False)

    plt.show()

Another example can be found in the tutorial:
:ref:`l-profile-ort-api`.

Graph Optimisations
===================

By default, :epkg:`onnxruntime` optimizes an ONNX graph as much
as it can. It removes every node it can, merges duplicates initializer,
fuses node into more complex node but more efficient such
*FusedMatMul* which deals with transposition as well.

There are four level of optimization and the final can be saved
on a disk to look at it.

::

    so = SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    # or GraphOptimizationLevel.ORT_ENABLE_BASIC
    # or GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # or GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = "to_save_the_optimized_onnx_file.onnx"

Example shows how to enable or disable optimizations on a simple
graph. The bigger the graph is, the more efficient they are.
See :ref:`benchmark-ort-onnx-graph-opt`.

Class :epkg:`InferenceSession` as any other class from
:epkg:`onnxruntime` cannot be pickled. Everything can
be created again from the ONNX file it loads. It also means
graph optimization are computed again. To speed up
the process, the optimized graph can be saved
and loaded with disabled optimization next time.
It can save the optimization time.
