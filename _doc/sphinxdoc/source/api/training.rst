
Training
========

.. contents::
    :local:

BaseEstimator
+++++++++++++

.. autosignature:: onnxcustom.training.optimizers.BaseEstimator
    :members:

LearningRate
++++++++++++

.. autosignature:: onnxcustom.training.sgd_learning_rate.LearningRateSGD
    :members:

.. autosignature:: onnxcustom.training.sgd_learning_rate.LearningRateSGDNesterov
    :members:

OrtGradientOptimizer
++++++++++++++++++++

.. autosignature:: onnxcustom.training.optimizers.OrtGradientOptimizer
    :members:

OrtGradientForwardBackward
++++++++++++++++++++++++++

.. autosignature:: onnxcustom.training.optimizers_partial.OrtGradientForwardBackwardOptimizer
    :members:

Helpers
+++++++

.. autosignature:: onnxcustom.utils.onnx_orttraining.add_loss_output

.. autosignature:: onnxcustom.utils.onnx_orttraining.get_train_initializer

Exceptions
++++++++++

.. autosignature:: onnxcustom.training.excs.ConvergenceError

Loss function
+++++++++++++

.. autosignature:: onnxcustom.utils.onnx_function.function_onnx_graph
