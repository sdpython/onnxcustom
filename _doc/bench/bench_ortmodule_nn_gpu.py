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
    py-spy record -o bench_ortmodule_nn_gpu.svg -r 20 -- python bench_ortmodule_nn_gpu.py --n_features 100 --hidden_layer_sizes "30,30"

Or the official profiler:

::

    python -m cProfile -o bench_ortmodule_nn_gpu.prof bench_ortmodule_nn_gpu.py --n_features 100 --hidden_layer_sizes "30,30"

The python can be profiled with :epkg:`pyinstrument`.

::

    python -m pyinstrument  --show-all -r html -o bench_ortmodule_nn_gpu.html bench_ortmodule_nn_gpu.py --n_features 100 --hidden_layer_sizes "30,30"

And with `nvprof` on GPU:

::

    nvprof -o bench_ortmodule_nn_gpu.nvprof python bench_ortmodule_nn_gpu.py --run_torch 0 --device cuda --opset 14

::

    nvprof -o bench_ortmodule_nn_gpu.sql python bench_ortmodule_nn_gpu.py --profile event --device cuda --opset 14

.. contents::
    :local:

A neural network with scikit-learn
++++++++++++++++++++++++++++++++++

"""
import warnings
from pprint import pprint
import time
import os
import numpy
from pandas import DataFrame
from onnxruntime import get_device
from cpyquickhelper.profiling import WithEventProfiler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from onnxruntime.training import ORTModule


def benchmark(N=1000, n_features=20, hidden_layer_sizes="26,25", max_iter=1000,
              learning_rate_init=1e-4, batch_size=100, run_torch=True,
              device='cpu', opset=12, profile='fct'):
    """
    Compares :epkg:`onnxruntime-training` to :epkg:`scikit-learn` for
    training. Training algorithm is SGD.

    :param N: number of observations to train on
    :param n_features: number of features
    :param hidden_layer_sizes: hidden layer sizes, comma separated values
    :param max_iter: number of iterations
    :param learning_rate_init: initial learning rate
    :param batch_size: batch size
    :param run_torch: train scikit-learn in the same condition (True) or
        just walk through one iterator with *scikit-learn*
    :param device: `'cpu'` or `'cuda'`
    :param opset: opset to choose for the conversion
    :param profile: 'fct' to use cProfile, 'event' to use WithEventProfiler
    """
    N = int(N)
    n_features = int(n_features)
    max_iter = int(max_iter)
    learning_rate_init = float(learning_rate_init)
    batch_size = int(batch_size)
    run_torch = run_torch in (1, True, '1', 'True')

    print("N=%d" % N)
    print("n_features=%d" % n_features)
    print(f"hidden_layer_sizes={hidden_layer_sizes!r}")
    print("max_iter=%d" % max_iter)
    print(f"learning_rate_init={learning_rate_init:f}")
    print("batch_size=%d" % batch_size)
    print(f"run_torch={run_torch!r}")
    print(f"opset={opset!r} (unused)")
    print(f"device={device!r}")
    print(f"profile={profile!r}")
    device0 = device
    device = torch.device(
        "cuda:0" if device in ('cuda', 'cuda:0', 'gpu') else "cpu")
    print(f"fixed device={device!r}")
    print('------------------')

    if not isinstance(hidden_layer_sizes, tuple):
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
    if device0 == 'cpu':
        nn.cpu()
    else:
        nn.cuda(device=device)
    print(f"n_parameters={len(list(nn.parameters()))}, n_layers={len(nn.hidden)}")
    for i, p in enumerate(nn.parameters()):
        print("  p[%d].shape=%r" % (i, p.shape))

    optimizer = torch.optim.SGD(nn.parameters(), lr=learning_rate_init)
    criterion = torch.nn.MSELoss(size_average=False)
    batch_no = len(X_train) // batch_size

    # training
    inputs = torch.tensor(
        X_train[:1], requires_grad=True, device=device)
    nn(inputs)

    def train_torch():
        for epoch in range(max_iter):
            running_loss = 0.0
            x, y = shuffle(X_train, y_train)
            for i in range(batch_no):
                start = i * batch_size
                end = start + batch_size
                inputs = torch.tensor(
                    x[start:end], requires_grad=True, device=device)
                labels = torch.tensor(
                    y[start:end], requires_grad=True, device=device)

                def step_torch():
                    optimizer.zero_grad()
                    outputs = nn(inputs)
                    loss = criterion(outputs, torch.unsqueeze(labels, dim=1))
                    loss.backward()
                    optimizer.step()
                    return loss

                loss = step_torch()
                running_loss += loss.item()
        return running_loss

    begin = time.perf_counter()
    if run_torch:
        if profile in ('cProfile', 'fct'):
            from pyquickhelper.pycode.profiling import profile
            running_loss, prof, _ = profile(train_torch, return_results=True)
            dur_torch = time.perf_counter() - begin
            name = f"{device0}.{os.path.split(__file__)[-1]}.tch.prof"
            prof.dump_stats(name)
        elif profile == 'event':

            def clean_name(x):
                return "/".join(x.replace("\\", "/").split('/')[-3:])

            prof = WithEventProfiler(size=10000000, clean_file_name=clean_name)
            with prof:
                running_loss = train_torch()
            dur_torch = time.perf_counter() - begin
            df = prof.report
            name = f"{device0}.{os.path.split(__file__)[-1]}.tch.csv"
            df.to_csv(name, index=False)
        else:
            running_loss = train_torch()
            dur_torch = time.perf_counter() - begin
    else:
        dur_torch = time.perf_counter() - begin

    if run_torch:
        print(f"time_torch={dur_torch!r}, running_loss={running_loss!r}")
        running_loss0 = running_loss
    else:
        running_loss0 = -1

    # ORTModule
    nn = Net(n_features, hidden_layer_sizes, 1)
    if device0 == 'cpu':
        nn.cpu()
    else:
        nn.cuda(device=device)

    nn_ort = ORTModule(nn)
    optimizer = torch.optim.SGD(nn_ort.parameters(), lr=learning_rate_init)
    criterion = torch.nn.MSELoss(size_average=False)

    # exclude onnx conversion
    inputs = torch.tensor(
        X_train[:1], requires_grad=True, device=device)
    nn_ort(inputs)

    def train_ort():
        for epoch in range(max_iter):
            running_loss = 0.0
            x, y = shuffle(X_train, y_train)
            for i in range(batch_no):
                start = i * batch_size
                end = start + batch_size
                inputs = torch.tensor(
                    x[start:end], requires_grad=True, device=device)
                labels = torch.tensor(
                    y[start:end], requires_grad=True, device=device)

                def step_ort():
                    optimizer.zero_grad()
                    outputs = nn_ort(inputs)
                    loss = criterion(outputs, torch.unsqueeze(labels, dim=1))
                    loss.backward()
                    optimizer.step()
                    return loss

                loss = step_ort()
                running_loss += loss.item()
        return running_loss

    begin = time.perf_counter()
    if profile in ('cProfile', 'fct'):
        from pyquickhelper.pycode.profiling import profile
        running_loss, prof, _ = profile(train_ort, return_results=True)
        dur_ort = time.perf_counter() - begin
        name = f"{device0}.{os.path.split(__file__)[-1]}.ort.prof"
        prof.dump_stats(name)
    elif profile == 'event':

        def clean_name(x):
            return "/".join(x.replace("\\", "/").split('/')[-3:])

        prof = WithEventProfiler(size=10000000, clean_file_name=clean_name)
        with prof:
            running_loss = train_ort()
        dur_ort = time.perf_counter() - begin
        df = prof.report
        name = f"{device0}.{os.path.split(__file__)[-1]}.ort.csv"
        df.to_csv(name, index=False)
    else:
        running_loss = train_ort()
        dur_ort = time.perf_counter() - begin

    print(f"time_torch={dur_torch!r}, running_loss={running_loss0!r}")
    print(f"time_ort={dur_ort!r}, last_trained_error={running_loss!r}")


if __name__ == "__main__":
    import fire
    fire.Fire(benchmark)
