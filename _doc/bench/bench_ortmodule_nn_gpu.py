"""

.. _l-orttraining-nn-benchmark:

Benchmark ORTModule on a neural network
=======================================

To make it work, you may need to run:

::

    python -c "from onnxruntime.training.ortmodule.torch_cpp_extensions import install as ortmodule_install;ortmodule_install.build_torch_cpp_extensions()"

You may profile the full example with on CPU with :epkg:`py-spy`:

::

    py-spy record -o bench_ortmodule_nn_gpu.svg -r 10 --native -- python bench_ortmodule_nn_gpu.py
    py-spy record -o bench_ortmodule_nn_gpu.svg -r 10 --native -- python bench_ortmodule_nn_gpu.py

And with `nvprof` on GPU:

::

    nvprof -o bench_ortmodule_nn_gpu.nvprof python bench_ortmodule_nn_gpu.py --run_skl 0 --device cuda --opset 14

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
from sklearn.utils import shuffle
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from onnxruntime.training import ORTModule


def benchmark(N=1000, n_features=20, hidden_layer_sizes="26,25", max_iter=1000,
              learning_rate_init=1e-4, batch_size=100, run_skl=True,
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
    print("hidden_layer_sizes=%s" % hidden_layer_sizes)
    print("max_iter=%d" % max_iter)
    print("learning_rate_init=%f" % learning_rate_init)
    print("batch_size=%d" % batch_size)
    print("run_skl=%r" % run_skl)
    print("opset=%r" % opset)
    print("device=%r" % device)
    device = torch.device("cuda:0" if device == 'cuda' else "cpu")
    print("fixed device=%r" % device)
    print('------------------')

    hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(",")))
    X, y = make_regression(N, n_features=n_features, bias=2)
    X = X.astype(numpy.float32)
    y = y.astype(numpy.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y)


    class Net(torch.nn.Module):
        def __init__(self, n_features, hidden, n_output):
            super(Net, self).__init__()
            self.hidden = []
            
            size = n_features
            for i, hid in enumerate(hidden_layer_sizes):
                self.hidden.append(torch.nn.Linear(size, hid))
                size = hid
                setattr(self, "hid%d" % i, self.hidden[-1])
            self.hidden.append(torch.nn.Linear(size, n_output))
            setattr(self, "predict", self.hidden[-1])

        def forward(self, x):
            for hid in self.hidden:
                x = hid(x)
                x = F.relu(x)
            return x

    nn = Net(n_features, hidden_layer_sizes, 1)
    print("n_parameters=%d, n_layers=%d" % (
        len(list(nn.parameters())), len(nn.hidden)))
    for i, p in enumerate(nn.parameters()):
        print("  p[%d].shape=%r" % (i, p.shape))
    
    optimizer = torch.optim.SGD(nn.parameters(), lr=learning_rate_init)
    criterion = torch.nn.MSELoss(size_average=False)
    batch_no = len(X_train) // batch_size

    # training

    begin = time.perf_counter()
    running_loss = 0.0
    for epoch in range(max_iter):
        X_train, y_train = shuffle(X_train, y_train)
        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            inputs = Variable(torch.FloatTensor(
                X_train[start:end], device=device))
            labels = Variable(torch.FloatTensor(
                y_train[start:end], device=device))

            optimizer.zero_grad()
            outputs = nn(inputs)
            loss = criterion(outputs, torch.unsqueeze(labels, dim=1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    dur_torch = time.perf_counter() - begin

    print("time_torch=%r, running_loss=%r" % (dur_torch, running_loss))
    running_loss0 = running_loss

    # ORTModule
    nn_ort = ORTModule(nn)
    optimizer = torch.optim.SGD(nn_ort.parameters(), lr=learning_rate_init)
    criterion = torch.nn.MSELoss(size_average=False)    
    
    begin = time.perf_counter()
    running_loss = 0.0
    for epoch in range(max_iter):
        X_train, y_train = shuffle(X_train, y_train)
        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            inputs = Variable(torch.FloatTensor(X_train[start:end]))
            labels = Variable(torch.FloatTensor(y_train[start:end]))

            optimizer.zero_grad()
            outputs = nn_ort(inputs)
            loss = criterion(outputs, torch.unsqueeze(labels, dim=1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    dur_ort = time.perf_counter() - begin

    print("time_torch=%r, running_loss=%r" % (dur_torch, running_loss0))
    print("time_ort=%r, last_trained_error=%r" % (dur_ort, running_loss))


if __name__ == "__main__":
    import fire
    fire.Fire(benchmark)
