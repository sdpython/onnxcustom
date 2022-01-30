
========
Training
========

There exists two APIs in :epkg:`onnxruntime`. One assumes
the loss function is part of the graph to derive, the other
one assumes the users provides the derivative of the loss
against the output of the graph. With the first API,
the weights are automatically updated. In the second API,
the users has to do it. It is more complex but gives more
freedom.

Both API are wrapped into two classes,
:ref:`l-api-prt-gradient-optimizer` for the first API,
:ref:`l-api-prt-gradient-optimizer-fw` for the second API.
Both classes make it easier to a user accustomed to
:epkg:`scikit-learn` API to train any graph with a
stochastic gradient descent algorithm.

.. contents::
    :local:

BaseEstimator
=============

Ancestor to both classes wrapping :epkg:`onnxruntime` API.

.. autosignature:: onnxcustom.training._base_estimator.BaseEstimator
    :members:

Exceptions
==========

.. autosignature:: onnxcustom.training.excs.ConvergenceError

.. autosignature:: onnxcustom.training.excs.EvaluationError

.. autosignature:: onnxcustom.training.excs.ProviderError

First API: loss part of the graph
=================================

Helpers
+++++++

Function `add_loss_output` adds a loss function to the graph
if this loss is part of the a predefined list. It may
be combination of L1, L2 losses and L1, L2 penalties.

.. autosignature:: onnxcustom.utils.orttraining_helper.add_loss_output

.. autosignature:: onnxcustom.utils.orttraining_helper.get_train_initializer

.. autosignature:: onnxcustom.utils.onnx_rewriter.onnx_rewrite_operator

.. _l-api-prt-gradient-optimizer:

OrtGradientOptimizer
++++++++++++++++++++

.. autosignature:: onnxcustom.training.optimizers.OrtGradientOptimizer
    :members:

Second API: loss part of the graph
==================================

ONNX
++++

Second API relies on class :epkg:`TrainingAgent`. It expects to find
the weight to train in alphabetical order. That's usual not the case.
The following function does not change the order but renames all
of them to fulfil that requirement.

.. autosignature:: onnxcustom.utils.onnx_helper.onnx_rename_weights

LearningPenalty
+++++++++++++++

.. autosignature:: onnxcustom.training.sgd_learning_penalty.NoLearningPenalty
    :members:

.. autosignature:: onnxcustom.training.sgd_learning_penalty.ElasticLearningPenalty
    :members:

LearningRate
++++++++++++

.. autosignature:: onnxcustom.training.sgd_learning_rate.LearningRateSGD
    :members:

.. autosignature:: onnxcustom.training.sgd_learning_rate.LearningRateSGDNesterov
    :members:

LearningLoss
++++++++++++

.. autosignature:: onnxcustom.training.sgd_learning_loss.AbsoluteLearningLoss
    :members:

.. autosignature:: onnxcustom.training.sgd_learning_loss.ElasticLearningLoss
    :members:

.. autosignature:: onnxcustom.training.sgd_learning_loss.NegLogLearningLoss
    :members:

.. autosignature:: onnxcustom.training.sgd_learning_loss.SquareLearningLoss
    :members:

Loss functions
++++++++++++++

.. autosignature:: onnxcustom.utils.onnx_function.function_onnx_graph

.. _l-api-prt-gradient-optimizer-fw:

OrtGradientForwardBackward
++++++++++++++++++++++++++

.. autosignature:: onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer
    :members:
