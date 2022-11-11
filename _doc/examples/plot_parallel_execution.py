"""
.. _l-plot-parallel-execution:

Multithreading with onnxruntime
===============================

.. index:: thread, parallel

Python implements multithreading but it is not in practice due to the GIL
(see :epkg:`Le GIL`). However, if most of the parallelized code is not creating
python object, this option becomes more interesting than creating several processes
trying to exchange data through sockets. :epkg:`onnxruntime` falls into that category.
For a big model such as a deeplearning model, this might be interesting.
This example verifies this scenario.

.. contents::
    :local:

A model
+++++++

Let's retrieve a not so big model.
"""
import os
import urllib.request
import numpy
from onnxruntime import InferenceSession


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


model_name = "squeezenet1.1-7.onnx"
url_name = ("https://github.com/onnx/models/raw/main/vision/"
            "classification/squeezenet/model")
url_name += "/" + model_name
download_file(url_name, model_name, 100000)

#############################################
# Measuring inference time
# ++++++++++++++++++++++++
#
# Let's create a random image.

sess = InferenceSession(model_name)
for i in sess.get_inputs():
    print(f"input {i}, name={i.name!r}, type={i.type}, shape={i.shape}")


rnd_img = numpy.random.rand((1, 3, 224, 224)).astype(numpy.float32)
