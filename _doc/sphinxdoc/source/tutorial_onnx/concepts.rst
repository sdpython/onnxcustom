
ONNX Concepts
=============

ONNX can be compared to a programming language specified
in mathematical functions. It defines all the necessary operations
a machine learning model needs to implement its inference function
with this langage. A linear regression could be represented
the following way:

::

    def onnx_linear_regressor(X):
        "ONNX code for a linear regression"
        return onnx.Add(onnx.MatMul(X, coefficients), bias)

.. index:: ONNX graph

This example is very similar to an expression a developper could
write in Python. It can be also represented as a graph which shows
step by step how to transform the features to get a prediction.
That's why a machine learning model implemented with ONNX is often
referenced as an **ONNX graph**.

.. image:: images/linreg1.png

ONNX aims at providing a common language any machine learning framework
can use to describe its models. The first scenario is to make it easier
to deploy a machine learning model in production. An ONNX interpretor
(or **runtime**) can be specifically implemented and optimized for this task
in the environment where it is deployed. With ONNX, it is possible
to build a unique process to deploy a model in production and independant
from the learning framework used to build the model.

.. contents::
    :local:

Input, Output, Node, Initializer
++++++++++++++++++++++++++++++++

Building an ONNX graph means implementing a function
with the ONNX language or more precisely the :ref:`l-onnx-operators`.
It is easier to read when there is one operator per line.
A linear regression would be written this way.

::

    x = onnx.input(0)
    a = onnx.input(1)
    c = onnx.input(2)

    ax = onnx.MatMul(a, x)
    axc = onnx.Add(ax, c)

    onnx.output(0) = axc

This code implements a function with the signature `f(x, a, c) -> axc`.
*a*, *a*, *c* are the **inputs**, *axc* is the **output**.
*ax* is an intermediate result.
Inputs and outputs are changing at each inference.
*MatMul* and *Add* are the **nodes**. They also have inputs and outputs.
A node has also a type, one of the operators in
:ref:`l-onnx-operators`. This graph was built with the example
in Section :ref:`l-onnx-linear-regression-onnx-api`.

The graph could also have an **initializer**. When an input
never changes such as the coefficients of the linear regression,
it be turned into a constant and stored into the graph.

::

    x = onnx.input(0)
    a = initializer
    c = initializer

    ax = onnx.MatMul(a, x)
    axc = onnx.Add(ax, c)

    onnx.output(0) = axc

Visually, this graph would look like this
(initializers are hidden). This graph was obtained with this
code :ref:`l-onnx-linear-regression-onnx-api-init`.

.. image:: images/linreg2.png

Serialization
+++++++++++++

Metadata
++++++++

List of available operators
+++++++++++++++++++++++++++

Supported Types
+++++++++++++++

ONNX is strongly typed and its definition does not support
implicit cast.

What is an opset version?
+++++++++++++++++++++++++

What is a domain?
+++++++++++++++++

Subgraphs
+++++++++

Extensibility
+++++++++++++

Shape Inference
+++++++++++++++

Tools
+++++

netron
