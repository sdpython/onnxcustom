"""

.. _l-orttraining-benchmark-torch:

Benchmark, comparison torch - forward-backward
==============================================

The benchmark compares the processing time between :epkg:`pytorch`
and :epkg:`onnxruntime-training` on a linear regression and a neural network.
This example starts from :ref:`l-orttraining-linreg-fwbw`
but uses :epkg:`pytorch` to replace the parts updating the gradients
and computing the error gradient. The training algorithm becomes:

.. image:: images/onnxfwbwtorch.png

Class :epkg:`TrainingAgent` (from :epkg:`onnxruntime-training`) is still
used and wrapped into :epkg:`ORTModule`. This script then
follows the same instructions as :ref:`l-orttraining-benchmark-fwbw`
to compare :epkg:`pytorch` only against :epkg:`pytorch` and
:epkg:`onnxruntime-training`.


.. contents::
    :local:

First comparison: neural network
++++++++++++++++++++++++++++++++

"""
import time
import numpy
from pandas import DataFrame
import torch
from onnxruntime import get_device
from onnxruntime.training.ortmodule import ORTModule
from pyquickhelper.pycode.profiling import profile, profile2graph
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression(2000, n_features=100, bias=2)
X = X.astype(numpy.float32)
y = y.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

########################################
# Common parameters and training algorithm
# ++++++++++++++++++++++++++++++++++++++++


def from_numpy(v, device=None, requires_grad=False):
    """
    Convers a numpy array into a torch array and
    sets *device* and *requires_grad*.
    """
    v = torch.from_numpy(v)
    if device is not None:
        v = v.to(device)
    v.requires_grad_(requires_grad)
    return v


###################################################
# Training, two functions with same code but it is easier
# to distinguish between in the profiling.

def train_model_torch(model, device, x, y, n_iter=100, learning_rate=1e-5,
                      profiler=None):
    model = model.to(device)
    x = from_numpy(x, requires_grad=True, device=device)
    y = from_numpy(y, requires_grad=True, device=device)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    for t in range(n_iter):

        def step_train_torch():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss

        loss = step_train_torch()
        losses.append(loss)
        if profiler is not None:
            profiler.step()

    return losses


def train_model_ort(model, device, x, y, n_iter=100, learning_rate=1e-5,
                    profiler=None):
    model = model.to(device)
    x = from_numpy(x, requires_grad=True, device=device)
    y = from_numpy(y, requires_grad=True, device=device)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    for t in range(n_iter):

        def step_train_ort():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss

        loss = step_train_ort()
        losses.append(loss)
        if profiler is not None:
            profiler.step()

    return losses

#################################
# Benchmark function


def benchmark(model_torch, model_ort, device, name, verbose=True):

    print("[benchmark] %s" % name)
    begin = time.perf_counter()
    losses = train_model_torch(
        model_torch, device, X_train, y_train, n_iter=200)
    duration_torch = time.perf_counter() - begin
    length_torch = len(losses)
    print("[benchmark] torch=%r iterations - %r seconds" % (
        length_torch, duration_torch))

    begin = time.perf_counter()
    losses = train_model_ort(model_ort, device, X_train, y_train, n_iter=200)
    duration_ort = time.perf_counter() - begin
    length_ort = len(losses)
    print("[benchmark] onxrt=%r iteration - %r seconds" % (
        length_ort, duration_ort))

    return dict(torch=duration_torch, ort=duration_ort, name=name,
                iter_torch=length_torch, iter_ort=length_ort)


class MLPNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(MLPNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 50)
        self.linear2 = torch.nn.Linear(50, 10)
        self.linear3 = torch.nn.Linear(10, D_out)

    def forward(self, x):
        o1 = torch.sigmoid(self.linear1(x))
        o2 = torch.sigmoid(self.linear2(o1))
        return self.linear3(o2)


d_in, d_out, N = X.shape[1], 1, X.shape[0]
model_torch = MLPNet(d_in, d_out)
model_ort = ORTModule(MLPNet(d_in, d_out))

device = torch.device('cpu')
benches = [benchmark(model_torch, model_ort, device, name='NN-CPU')]

######################################
# Profiling
# +++++++++


def clean_name(text):
    pos = text.find('onnxruntime')
    if pos >= 0:
        return text[pos:]
    pos = text.find('onnxcustom')
    if pos >= 0:
        return text[pos:]
    pos = text.find('torch')
    if pos >= 0:
        return text[pos:]
    pos = text.find('site-packages')
    if pos >= 0:
        return text[pos:]
    return text


ps = profile(lambda: benchmark(
    model_torch, model_ort, device, name='LR-CPU'))[0]
root, nodes = profile2graph(ps, clean_text=clean_name)
text = root.to_text()
print(text)

######################################
# if GPU is available
# +++++++++++++++++++

if get_device().upper() == 'GPU':

    device = torch.device('cuda:0')
    benches.append(benchmark(model_torch, model_ort, device, name='LR-GPU'))

######################################
# Linear Regression
# +++++++++++++++++


class LinearRegressionNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(LinearRegressionNet, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        return self.linear(x)


d_in, d_out, N = X.shape[1], 1, X.shape[0]
model_torch = LinearRegressionNet(d_in, d_out)
model_ort = ORTModule(LinearRegressionNet(d_in, d_out))

device = torch.device('cpu')
benches.append(benchmark(model_torch, model_ort, device, name='LR-CPU'))


if get_device().upper() == 'GPU':

    device = torch.device('cuda:0')
    benches.append(benchmark(model_torch, model_ort, device, name='LR-GPU'))

    ######################################
    # GPU profiling
    # +++++++++++++

    if get_device().upper() == 'GPU':
        ps = profile(lambda: benchmark(
            model_torch, model_ort, device, name='LR-GPU'))[0]
        root, nodes = profile2graph(ps, clean_text=clean_name)
        text = root.to_text()
        print(text)

######################################
# Graphs
# ++++++
#
# Dataframe first.

df = DataFrame(benches).set_index('name')
df

#######################################
# text output

print(df)

#######################################
# Graphs.

print(df.columns)
ax = df[['torch', 'ort']].plot.bar(title="Processing time")
ax.tick_params(axis='x', rotation=30)

# import matplotlib.pyplot as plt
# plt.show()
