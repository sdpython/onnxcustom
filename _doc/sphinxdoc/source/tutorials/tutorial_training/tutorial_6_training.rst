
.. _l-full-training:

Full Training with OrtGradientOptimizer
=======================================

.. contents::
    :local:

Design
++++++

:epkg:`onnxruntime` was initially designed to speed up inference
and deployment but it can also be used to train a model.
It builds a graph equivalent to the gradient function
also based on onnx operators and specific gradient operators.
Initializers are weights that can be trained. The gradient graph
has as many as outputs as initializers.

:class:`OrtGradientOptimizer
<onnxcustom.training.optimizers.OrtGradientOptimizer>` wraps
class :epkg:`TrainingSession` from :epkg:`onnxruntime-training`.
It starts with one model converted into ONNX graph.
A loss must be added to this graph. Then class :epkg:`TrainingSession`
is able to compute another ONNX graph equivalent to the gradient
of the loss against the weights defined by intializers.

The first ONNX graph implements a function *Y=f(W, X)*.
Then function :func:`add_loss_output
<onnxcustom.utils.orttraining_helper.add_loss_output>`
adds a loss to define a graph *loss, Y=loss(f(W, X), W, expected_Y)*.
This same function is able to add the necessary nodes to compute
L1 and L2 losses or a combination of both, a L1 or L2 regularizations
or a combination of both. Assuming the user was able to create
an an ONNX graph, he would add *0.1 L1 loss + 0.9 L2 loss*
and a L2 regularization on the coefficients by calling :func:`add_loss_output
<onnxcustom.utils.orttraining_helper.add_loss_output>`
like that:

::

    onx_loss = add_loss_output(
        onx, weight_name='weight', score_name='elastic',
        l1_weight=0.1, l2_weight=0.9,
        penalty={'coef': {'l2': 0.01}})

An instance of class :class:`OrtGradientOptimizer
<onnxcustom.training.optimizers.OrtGradientOptimizer>` is
initialized:

::

    train_session = OrtGradientOptimizer(
        onx_loss, ['intercept', 'coef'], learning_rate=1e-3)

And then trained:

::

    train_session.fit(X_train, y_train, w_train)

Coefficients can be retrieved like the following:

::

    state_tensors = train_session.get_state()

And train losses:

::

    losses = train_session.train_losses_

This design does not allow any training with momentum,
keeping an accumulator for gradients yet.
The class does not expose all the possibilies implemented in
:epkg:`onnxruntime-training`.
Next examples show that in practice.

Examples
++++++++

The first example compares a linear regression trained with
:epkg:`scikit-learn` and another one trained with
:epkg:`onnxruntime-training`.

The two next examples explains in details how the training
with :epkg:`onnxruntime-training`. They dig into class
:class:`OrtGradientOptimizer
<onnxcustom.training.optimizers.OrtGradientOptimizer>`.
It leverages class :epkg:`TrainingSession` from :epkg:`onnxruntime-training`.
This one assumes the loss function is part of the graph to train.
It takes care to the weight updating as well.

The fourth example replicates what was done with the linear regression
but with a neural network built by :epkg:`scikit-learn`.
It trains the network on CPU or GPU
if it is available. The last example benchmarks the different
approaches.

.. toctree::
    :maxdepth: 1

    ../../gyexamples/plot_orttraining_linear_regression
    ../../gyexamples/plot_orttraining_linear_regression_cpu
    ../../gyexamples/plot_orttraining_linear_regression_gpu
    ../../gyexamples/plot_orttraining_nn_gpu
    ../../gyexamples/plot_orttraining_benchmark
