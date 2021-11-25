
Training
========

:epkg:`onnxruntime` was initially designed to speed up inference
and deployment but it can also be used to train a model.
It builds a graph equivalent to the gradient function
also based on onnx operators and specific gradient operators.
Initializers are weights that can be trained. The gradient graph
has as many as outputs as initializers.
The first example compares a linear regression trained with
:epkg:`scikit-learn` and another one trained with
:epkg:`onnxruntime-training`.

The two next examples explains in details how the training
with :epkg:`onnxruntime-training`. They dig into class
:class:`OrtGradientOptimizer
<onnxcustom.training.optimizers.OrtGradientOptimizer>`.

The fourth example replicates what was done with the linear regression
but with a neural network built by :epkg:`scikit-learn`.
It trains the network on CPU or GPU
if it is available. The last example benchmarks the different
approaches.

.. toctree::
    :maxdepth: 1

    ../gyexamples/plot_orttraining_linear_regression
    ../gyexamples/plot_orttraining_linear_regression_cpu
    ../gyexamples/plot_orttraining_linear_regression_gpu
    ../gyexamples/plot_orttraining_nn_gpu
    ../gyexamples/plot_orttraining_benchmark
