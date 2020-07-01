"""
What is the opset number?
=========================

Every library is versioned. :epkg:`scikit-learn` may change
the implementation of a specific model. That happens
for example with the `SVC <https://scikit-learn.org/stable/
modules/generated/sklearn.svm.SVC.html>`_ model where
the parameter *break_ties* was added in 0.22. :epkg:`ONNX`
does also have a version called *opset number*.
Operator *ArgMin* was added in opset 1 and changed in opset
11, 12, 13. Sometimes, it is updated to extend the list
of types it supports, sometimes, it moves a parameter
into the input list. The runtime used to deploy the model
does not implement a new version, in that case, a model
must be converted by usually using the most recent opset
supported by the runtime, we call that opset the
*targeted opset*. An ONNX graph only contains
one unique opset, every node must be described following
the specifications defined by the latest opset below the
targeted opset.

*to be continued*
"""
