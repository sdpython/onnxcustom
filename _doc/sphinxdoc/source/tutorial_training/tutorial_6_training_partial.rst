
Partial Training
================

Section :ref:`l-full-training` introduces a class able a while
ONNX graph. :epkg:`onnxruntime-training` handles the computation
of the loss, the gradient, it updates the weights as well.
This design does not work when ONNX graph only plays a part
in the model and is not the whole model. A deep neural network could
be composed with a first layer from :epkg:`torch`, a second one from
ONNX, and be trained by a gradient descent implemented in python.

Partial training is another way to train an ONNX model. It can be trained
as a standalone ONNX graph or be integrated in a :epkg:`torch` model or any
framework implementing *forward* and *backward* mechanism.
Next example introduced how this is done with ONNX.

.. toctree::
    :maxdepth: 1

    ../gyexamples/plot_orttraining_linear_regression_fwbw
    ../gyexamples/plot_orttraining_nn_gpu_fwbw
    ../gyexamples/plot_orttraining_nn_gpu_fwbw_nesterov
    ../gyexamples/plot_orttraining_benchmark_fwbw
