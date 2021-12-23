
=======================
onnxruntime with Python
=======================

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

Some informations about the graph can be retrieve
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

Session and Running options
===========================

Providers
=========

Graph Optimisations
===================
