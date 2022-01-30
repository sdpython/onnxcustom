
=========================
Training with onnxruntime
=========================

:epkg:`onnxruntime` offers the possibility to compute
a gradient. Then with some extra lines, it is possible
to implement a gradient descent.

Training capabilities are part of the same repository
but released under a different package :epkg:`onnxruntime-training`.
It is not an extension, it replaces :epkg:`onnxruntime`
and has the same import name. It can be built with
different compilation settings or downloaded from pypi.
There are two versions to keep a low size for the version
which only does inference.

Two training API are available. The first one assumes the loss
is part of the graph to train. It can be trained as a whole.
The second API assumes the graph is only a piece or
a layer in a model trained by another framework or at
least a logic which updates the weights. This mechanism
is convenient when a model is trained with :epkg:`pytorch`.

.. contents::
    :local:

.. _l-orttraining-first-api:

First API: TrainingSession
==========================

:epkg:`TrainingSession` is used by class
:class:`OrtGradientOptimizer
<onnxcustom.training.optimizers.OrtGradientOptimizer>` in example
:ref:`l-orttraining-linreg` to show how it could be wrappped
to train a model. Example :ref:`l-orttraining-linreg-cpu` digs
into the details of the implementation. It goes through the following
steps:

* express the loss with ONNX operators
* select all initializers to train
* fill an instance of :epkg:`TrainingParameters`
* create an instance of :epkg:`TrainingSession`

That's what method :meth:`OrtGradientOptimizer._create_training_session
<onnxcustom.training.optimizers.OrtGradientOptimizer._create_training_session>`
does. It does not implement a training algorithm, only an iteration
- forward + backward - with the expected label, the learning rate and the features
as inputs. The class updates its weights. When the training ends, the user
must collect the updated weights and create a new ONNX file with the
optimized weights.

:epkg:`onnxruntime-training` does not implement loss functions.
That must be done independently. That's what function
:func:`onnxcustom.utils.orttraining_helper.add_loss_output` does.
It implements a couple of usual losses in ONNX.
Another function :func:`onnxcustom.utils.orttraining_helper.get_train_initializer`
guesses all the coefficients of an ONNX graph if the user does not specify any.
Another common use not implemented in :epkg:`onnxruntime-training`.

GPU is no different. It changes the syntax because data has to
be moved on this device first. Example :ref:`l-orttraining-linreg-gpu`
adapts previous example to this configuration.
Finally, a last example compares this approach against
:epkg:`scikit-learn` in the same conditions.

.. _l-orttraining-s-api:

Second API: TrainingAgent
=========================

:epkg:`TrainingAgent` is used by class
:class:`OrtGradientForwardBackwardOptimizer
<onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer>`
to train the same model. The training is split into the
forward step, the backward step (gradient computation), the weight
updating step. :epkg:`TrainingAgent` implement forward and backward.
Everything else must be explicitely implemented outside of this class
or be taken care of by an existing framework such as this one
or :epkg:`pytorch`. First, forward, backward with :epkg:`TrainingAgent`.
To build it, the following steps are needed:

* fill an instance of :epkg:`OrtModuleGraphBuilderConfiguration`
* create the training graph with :epkg:`OrtModuleGraphBuilder`
* retrieve the training graph
* create an instance of :epkg:`InferenceSession` with this graph
* create an instance of :epkg:`TrainingAgent`

That's what method :meth:`OrtGradientForwardBackward._create_onnx_graphs
<onnxcustom.training.ortgradient.OrtGradientForwardBackward._create_onnx_graphs>`
does. Forward and backward steps must be called separately.
It is not trivial to guess how to call them (a forward step can be
called to predict or to train if followed by a backward step).
Class :class:`OrtGradientForwardBackwardFunction
<onnxcustom.training.ortgradient.OrtGradientForwardBackwardFunction>`
implements those two steps with the proper API. Next lines gives an
idea on how it can be done. First the forward step.

::

    def forward(self, inputs, training=False):
        forward_inputs = cls.input_to_ort(
            inputs, cls._devices, cls._debug)

        if training:
            forward_outputs = OrtValueVector()
            state = PartialGraphExecutionState()
            self.states_.append(state)
            cls._training_agent.run_forward(
                forward_inputs, forward_outputs, state, cls._cache)
            return forward_outputs
        else:
            iobinding = SessionIOBinding(cls._sess_eval._sess)
            for name, inp in zip(
                    cls._grad_input_names, forward_inputs):
                iobinding.bind_ortvalue_input(name, inp)

            for name, dev in zip(
                    cls._output_names, cls._fw_no_grad_output_device_info):
                iobinding.bind_output(name, dev)

            cls._sess_eval._sess.run_with_iobinding(
                iobinding, cls._run_options)
            return iobinding.get_outputs()

Then the backward step.

::

    def backward(self, grad_outputs):
        cls = self.__class__
        inputs = self.saved_tensors
        state = self.states_.pop()
        backward_inputs = cls.input_to_ort(
            grad_outputs, cls._bw_outputs_device_info, cls._debug)

        backward_outputs = OrtValueVector()
        cls._training_agent.run_backward(
            backward_inputs, backward_outputs, state)
        return backward_outputs

The API implemented by class :epkg:`TrainingAgent` does not
use named inputs, only a list of inputs, the features followed
by the current weights. Initializers must be be given
names in alphabetical order to avoid any confusion with that API.

:ref:`l-orttraining-linreg-fwbw` changes the previous example
to use class :class:`OrtGradientForwardBackwardOptimizer
<onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer>`
explains the details of the implementation. This example is the best
place to continue if using the raw API of :epkg:`onnxruntime-training`
is the goal. Then the same
example is changed to use GPU: :ref:`l-orttraining-nn-gpu-fwbw`.
And finally a benchmark to compare this approach with
:epkg:`scikit-learn`: :ref:`l-orttraining-benchmark-fwbw`.
Beside forward and backard, the training needs three elements
to be complete.

* **a loss**: a square loss for example :class:`SquareLearningLoss
  <onnxcustom.training.sgd_learning_loss.SquareLearningLoss>`
  but it could be :class:`ElasticLearningPenalty
  <onnxcustom.training.sgd_learning_penalty.ElasticLearningPenalty>`.
* **a way to update the weight**: a simple learning rate for a stockastic
  gradient descent :class:`LearningRateSGD
  <onnxcustom.training.sgd_learning_rate.LearningRateSGD>` or
  something more complex such as :class:`LearningRateSGDNesterov
  <onnxcustom.training.sgd_learning_rate.LearningRateSGDNesterov>`.
* **a regularization applied to the weight**, it could be seen as an extension
  of the loss but this design seemed more simple as it does not mix
  the gradient applied to the output and the gradient due to the
  regularization, the most simple regularization is no regularization with
  :class:`NoLearningPenalty
  <onnxcustom.training.sgd_learning_penalty.NoLearningPenalty>`,
  but it could be L1 or L2 penalty as well with :class:`ElasticLearningPenalty
  <onnxcustom.training.sgd_learning_penalty.ElasticLearningPenalty>`.

These parts can easily be replaced by the same pieces
implemented in :epkg:`pytorch`. That's what wrapper
class :epkg:`ORTModule` offers except it starts from a :epkg:`pytorch`
model then converted into ONNX. That's what shows example
:ref:`l-orttraining-benchmark-torch`. Class :class:`OrtGradientForwardBackwardOptimizer
<onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer>`
directly starts with the ONNX graph and adds the pieces not implemented
in :epkg:`onnxruntime-training`.
