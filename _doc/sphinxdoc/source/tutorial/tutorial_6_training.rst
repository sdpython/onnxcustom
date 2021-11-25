
Training
========

:epkg:`onnxruntime` was initially designed to speed up inference
and deployment but it can also be used to train a model.
It builds a graph equivalent to the gradient function
also based on onnx operators and specific gradient operators.
Initializers are weights that can be trained. The gradient graph
has as many as outputs as initializers.
The first example explains how to train a linear model
with onnxruntime. The second one modifies the first example
to do it with GPU. The third example extends that experiment
with a neural network built by :epkg:`scikit-learn`.

.. toctree::
    :maxdepth: 1

    ../gyexamples/plot_orttraining_linear_regression
    ../gyexamples/plot_orttraining_linear_regression_gpu
    ../gyexamples/plot_orttraining_nn_gpu
