"""
.. _l-plot-parallel-execution-big-models:

==============================================
Multithreading with onnxruntime and big models
==============================================

.. index:: thread, parallel, onnxruntime, gpu, big models

Example :ref:`l-plot-parallel-execution` shows that parallelizing the inference
over multiple GPUs on the same machine is worth doing it. However,
this may not be possible when the model is too big to hold in the
memory of a single GPU. In that case, we need to split the model
and have each of the GPU run a piece of it.
The strategy implemented in this example consists in dividing the model
layers into consecutives pieces and push them on separate GPU.
Let's assume a random network has two layers L1 and L2 roughly of the same size,
GPU 1 will host L1, GPU 2 does the same with L1. A batch size contains 2 images.
Their inference can decomposed the following way:

* :math:`t_1`: image 1 is copied on GPU 1
* :math:`t_2`: L1 is processed
* :math:`t_3`: output of L1 is copied to GPU 2, image 2 is copied to GPU 1
* :math:`t_4`: L1, L2 are processed.
* :math:`t_5`: output of L1 is copied to GPU 2, output of L2 is copied to CPU
* :math:`t_6`: L2 is processed
* :math:`t_7`: output of L2 is copied to CPU

This works if the copy accross GPU does not take too much time.
The improvment should be even better for a longer batch.

.. contents::
    :local:

A model
=======

Let's retrieve a not so big model. They are taken from the
`ONNX Model Zoo <https://github.com/onnx/models>`_ or can even be custom.
"""
import gc
import multiprocessing
import os
import pickle
from pprint import pprint
import urllib.request
import threading
import time
import sys
import tqdm
import numpy
import pandas
import onnx
import torch.cuda
from onnxruntime import InferenceSession, get_all_providers
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtDevice as C_OrtDevice, OrtValue as C_OrtValue)
from onnxcustom.utils.onnx_split import split_onnx


def download_file(url, name, min_size):
    if not os.path.exists(name):
        print(f"download '{url}'")
        with urllib.request.urlopen(url) as u:
            content = u.read()
        if len(content) < min_size:
            raise RuntimeError(
                f"Unable to download '{url}' due to\n{content}")
        print(f"downloaded {len(content)} bytes.")
        with open(name, "wb") as f:
            f.write(content)
    else:
        print(f"'{name}' already downloaded")


small = "custom" if "custom" in sys.argv else "small"
if small == "custom":
    model_name = "gpt2.onnx"
    url_name = None
    maxN = 5
    stepN = 1
    repN = 4
    big_model = True
elif small:
    model_name = "mobilenetv2-10.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/mobilenet/model")
    maxN = 21
    stepN = 2
    repN = 4
    big_model = False
else:
    model_name = "resnet18-v1-7.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/resnet/model")
    maxN = 21
    stepN = 2
    repN = 4
    big_model = False

if url_name is not None:
    url_name += "/" + model_name
    download_file(url_name, model_name, 100000)


########################################
# GPU
# ===
#
# Let's check first if it is possible.

has_cuda = "CUDAExecutionProvider" in get_all_providers()
if not has_cuda:
    print(f"No CUDA provider was detected in {get_all_providers()}.")

n_gpus = torch.cuda.device_count() if has_cuda else 0
if n_gpus == 0:
    print("No GPU or one GPU was detected.")
elif n_gpus == 1:
    print("1 GPU was detected.")
else:
    print(f"{n_gpus} GPUs were detected.")

#############################################
# Split the model
# ===============
#
# It is an ONNX graph. There is no notion of layers.
# The function :func:`split_onnx <onnxcustom.utils.onnx_split.split_onnx>`
# first detects possible cutting points (breaking the connexity of the graph)
# Then it is just finding the best cutting points to split the model into
# pieces of roughly the same size.

with open(model_name, "rb") as f:
    model = onnx.load(f)

n_parts = max(n_gpus, 2)
pieces = split_onnx(model, n_parts, verbose=1)

for i, piece in enumerate(pieces):
    with open(f"piece-{os.path.splitext(model_name)[0]}-{i}.onnx", "wb") as f:
        f.write(piece.SerializeToString())
