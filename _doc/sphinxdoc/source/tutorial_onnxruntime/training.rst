
=========================
Training with onnxruntime
=========================

:epkg:`onnxruntime` offers the possibility to compute
a gradient. Then we some extra lines, it is possible
to implement a gradient descent.

Training capabilities were released in another package
:epkg:`onnxruntime-training`. It is not an extension,
it replaces :epkg:`onnxruntime` and has the same import
name. It can be built with different compilation settings
or downloaded from pypi. There are two versions to keep
a low size for the version which only does inference.

Two API are available. The first one assumes the loss
is part of the graph to train. It can be trained as a whole.
The second API assumes the graph is only a piece or
a layer in a model trained by another framework or at
least a logic which updates the weights. This mechanism
is convenient when a model is trained with :epkg:`pytorch`.

.. contents::
    :local:

First API: TrainingSession
==========================

:epkg:`TrainingSession` is used by class
:class:`OrtGradientOptimizer
<onnxcustom.training.optimizers.OrtGradientOptimizer>` in example
:ref:`l-orttraining-linreg` to show how it could be wrappped
to train a model. Example :ref:`l-orttraining-linreg-cpu` digs
into the details of the implementation.

GPU is no different. It changes the syntax because data has to
be moved on this device first. Example :ref:`l-orttraining-linreg-gpu`
adapts previous example to this configuration.

Finally, a last example compares this approach against
:epkg:`scikit-learn` in the same conditions.

Second API: TrainingAgent
=========================

:epkg:`TrainingAgent` is used by class
:class:`OrtGradientForwardBackwardOptimizer
<onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer>`
to train the same model. It splits the training into the
forward step, the backward (gradient computation) and the weight
updates. All must be explicitely implemented outside of this class
or be taken care of by an existing framework.

:ref:`l-orttraining-linreg-fwbw` changes the previous example
to use class
:class:`OrtGradientForwardBackwardOptimizer
<onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer>`
explains the details of the implementation. Then the same
example is changed to use GPU: :ref:`l-orttraining-nn-gpu-fwbw`.
And finally a benchmark to compare this approach with
:epkg:`scikit-learn`: :ref:`l-orttraining-benchmark-fwbw`.

That approach is also to interact with :epkg:`pytorch`. The logic
explained above and much more than that is implemented in
class :epkg:`ORTModule`. That's what shows example
:ref:`l-orttraining-benchmark-torch`.
