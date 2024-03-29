"""

.. _l-orttraining-nn-benchmark-fwbw:

Benchmark onnxruntime-training on a neural network, forward, backward
=====================================================================

You may profile the full example with on CPU with :epkg:`py-spy`:

::

    py-spy record -o bench_orttraining_nn_gpu_fwbw.svg -r 10 --native -- python bench_orttraining_nn_gpu_fwbw.py

And with `nvprof` on GPU:

::

    nvprof -o bench_orttraining_nn_gpu.nvprof python bench_orttraining_nn_gpu_fwbw.py --run_skl 0 --device cuda --opset 14

.. contents::
    :local:

A neural network with scikit-learn
++++++++++++++++++++++++++++++++++

"""
import warnings
from pprint import pprint
import time
import numpy
from pandas import DataFrame
from onnxruntime import get_device
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from mlprodict.onnx_conv import to_onnx
from onnxcustom.utils.orttraining_helper import get_train_initializer
from onnxcustom.utils.onnx_helper import onnx_rename_weights
from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer


def benchmark(N=1000, n_features=100, hidden_layer_sizes="50,50", max_iter=500,
              learning_rate_init=1e-8, batch_size=15, run_skl=True,
              device='cpu', opset=14):
    """
    Compares :epkg:`onnxruntime-training` to :epkg:`scikit-learn` for
    training. Training algorithm is SGD.

    :param N: number of observations to train on
    :param n_features: number of features
    :param hidden_layer_sizes: hidden layer sizes, comma separated values
    :param max_iter: number of iterations
    :param learning_rate_init: initial learning rate
    :param batch_size: batch size
    :param run_skl: train scikit-learn in the same condition (True) or
        just walk through one iterator with *scikit-learn*
    :param device: `'cpu'` or `'cuda'`
    :param opset: opset to choose for the conversion
    """
    N = int(N)
    n_features = int(n_features)
    max_iter = int(max_iter)
    learning_rate_init = float(learning_rate_init)
    batch_size = int(batch_size)
    run_skl = run_skl in (1, True, '1', 'True')

    print("N=%d" % N)
    print("n_features=%d" % n_features)
    print(f"hidden_layer_sizes={hidden_layer_sizes!r}")
    print("max_iter=%d" % max_iter)
    print(f"learning_rate_init={learning_rate_init:f}")
    print("batch_size=%d" % batch_size)
    print(f"run_skl={run_skl!r}")
    print(f"opset={opset!r}")
    print(f"device={device!r}")
    print('------------------')

    if not isinstance(hidden_layer_sizes, tuple):
        hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(",")))
    X, y = make_regression(N, n_features=n_features, bias=2)
    X = X.astype(numpy.float32)
    y = y.astype(numpy.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                      max_iter=max_iter if run_skl else 1,
                      solver='sgd', learning_rate_init=learning_rate_init,
                      n_iter_no_change=max_iter, batch_size=batch_size,
                      alpha=0, nesterovs_momentum=False, momentum=0,
                      learning_rate="invscaling")

    begin = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nn.fit(X_train, y_train)
    dur_skl = time.perf_counter() - begin

    print("time_skl=%r, mean_squared_error=%r" % (
        dur_skl, mean_squared_error(y_train, nn.predict(X_train))))

    # conversion to ONNX
    onx = to_onnx(nn, X_train[:1].astype(numpy.float32), target_opset=opset)
    onx = onnx_rename_weights(onx)

    # list of weights
    weights = get_train_initializer(onx)
    print('weights:', list(sorted(weights)))

    # training
    print(f"device={device!r} get_device()={get_device()!r}")

    #######################################
    # The training session.

    train_session = OrtGradientForwardBackwardOptimizer(
        onx, list(weights), device=device, verbose=0,
        learning_rate=learning_rate_init,
        warm_start=False, max_iter=max_iter, batch_size=batch_size)

    begin = time.perf_counter()
    train_session.fit(X, y)
    dur_ort = time.perf_counter() - begin
    print("time_skl=%r, mean_squared_error=%r" % (
        dur_skl, mean_squared_error(y_train, nn.predict(X_train))))
    print("time_ort=%r, last_trained_error=%r" % (
        dur_ort, train_session.train_losses_[-1]))


if __name__ == "__main__":
    import fire
    fire.Fire(benchmark)
