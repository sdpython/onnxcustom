
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

.. image:: images/linreg.png

ONNX aims at providing a common language any machine learning framework
can use to describe its models. The first scenario is to make it easier
to deploy a machine learning model in production. An ONNX interpretor
(or **runtime**) can be specifically implemented and optimized for this task
in the environment where it is deployed. With ONNX, it is possible
to build a unique process to deploy a model in production and independant
from the learning framework used to build the model.

.. contents::
    :local:

A simple example: a linear regression
+++++++++++++++++++++++++++++++++++++

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
