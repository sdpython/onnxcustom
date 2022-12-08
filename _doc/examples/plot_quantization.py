"""
.. _l-plot-quantization:

==============================
Quantization with onnxruntime
=============================

.. index:: quantization, onnxruntime

Quantization aims at reducing the model size but it does
compute the output at a lower precision too.
The static quantization estimates the best quantization parameters
for every observation in a dataset. The dynamic quantization
estimates these parameters for every observation at inference time.
Let's see the differences
(see alse `Quantize ONNX Models
<https://onnxruntime.ai/docs/performance/quantization.html>`_).


.. contents::
    :local:

A model
=======

Let's retrieve a not so big model. They are taken from the
`ONNX Model Zoo <https://github.com/onnx/models>`_ or can even be custom.
"""
import os
import urllib.request
import time
import tqdm
import numpy
import onnx
import pandas
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from onnxruntime.quantization.quantize import quantize_dynamic, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process


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


small = "small"
if small:
    model_name = "mobilenetv2-12.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/mobilenet/model")
else:
    model_name = "resnet50-v1-12.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/resnet/model")

if url_name is not None:
    url_name += "/" + model_name
    download_file(url_name, model_name, 100000)

#############################
# Inputs and outputs.

sess_full = InferenceSession(model_name, providers=["CPUExecutionProvider"])

for i in sess_full.get_inputs():
    print(f"input {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    input_name = i.name
    input_shape = list(i.shape)
    if input_shape[0] in [None, "batch_size", "N"]:
        input_shape[0] = 1

output_name = None
for i in sess_full.get_outputs():
    print(f"output {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    if output_name is None:
        output_name = i.name

print(f"input_name={input_name!r}, output_name={output_name!r}")

################################
# We build random data.

maxN = 50
imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
        for i in range(maxN)]

experiments = []

#############################################
# Static Quantization
# ===================
#
# This quantization estimates the best quantization parameters
# (scale and bias) to minimize an error compare to the original
# model. It requires data.


class DataReader(CalibrationDataReader):
    def __init__(self, input_name, imgs):
        self.input_name = input_name
        self.data = imgs
        self.pos = -1

    def get_next(self):
        if self.pos >= len(self.data) - 1:
            return None
        self.pos += 1
        return {self.input_name: self.data[self.pos]}

    def rewind(self):
        self.pos = -1

##############################
# Runs the quantization.


quantize_name = model_name + ".qdq.onnx"


quantize_static(model_name,
                quantize_name,
                calibration_data_reader=DataReader(input_name, imgs),
                quant_format=QuantFormat.QDQ)

####################################
# Compares the size.

with open(model_name, "rb") as f:
    model_onnx = onnx.load(f)
with open(quantize_name, "rb") as f:
    quant_onnx = onnx.load(f)

model_onnx_bytes = model_onnx.SerializeToString()
quant_onnx_bytes = quant_onnx.SerializeToString()

print(f"Model size: {len(model_onnx_bytes)} and "
      f"quantized: {len(quant_onnx_bytes)}, "
      f"ratio={len(quant_onnx_bytes) / len(model_onnx_bytes)}.")

##########################################
# Let's measure the dIscrepancies.


def compare_with(sess_full, imgs, quantize_name):
    sess = InferenceSession(quantize_name, providers=["CPUExecutionProvider"])

    mean_diff = 0
    mean_max = 0
    time_full = 0
    time_quant = 0
    disa = 0

    for img in tqdm.tqdm(imgs):
        feeds = {input_name: img}

        begin = time.perf_counter()
        full = sess_full.run(None, feeds)
        time_full += time.perf_counter() - begin

        begin = time.perf_counter()
        quant = sess.run(None, feeds)
        time_quant += time.perf_counter() - begin

        diff = numpy.abs(full[0] - quant[0]).ravel()
        mean_max += numpy.abs(full[0].ravel().max() - quant[0].ravel().max())
        mean_diff += diff.mean()
        if full[0].argmax() != quant[0].argmax():
            disa += 1

    mean_diff /= len(imgs)
    mean_max /= len(imgs)
    time_full /= len(imgs)
    time_quant /= len(imgs)
    return dict(mean_diff=mean_diff, mean_max=mean_max,
                time_full=time_full, time_quant=time_quant,
                disagree=disa / len(imgs),
                ratio=time_quant / time_full)


res = compare_with(sess_full, imgs, quantize_name)
res["name"] = "static"
experiments.append(res)
print(f"Discrepancies: mean={res['mean_diff']:.2f}, "
      f"mean_max={res['mean_max']:.2f}, "
      f"times {res['time_full']} -> {res['time_quant']}, "
      f"disagreement={res['disagree']:.2f}")
res

#######################################
# With preprocessing
# ==================

preprocessed_name = model_name + ".pre.onnx"

quant_pre_process(model_name, preprocessed_name)

#################################
# And quantization again.

quantize_static(preprocessed_name,
                quantize_name,
                calibration_data_reader=DataReader(input_name, imgs),
                quant_format=QuantFormat.QDQ)

res = compare_with(sess_full, imgs, quantize_name)
res["name"] = "static-pre"
experiments.append(res)
print(f"Discrepancies: mean={res['mean_diff']:.2f}, "
      f"mean_max={res['mean_max']:.2f}, "
      f"times {res['time_full']} -> {res['time_quant']}, "
      f"disagreement={res['disagree']:.2f}")
res

#########################################
# Dynamic quantization
# ====================

quantize_name = model_name + ".qdq.dyn.onnx"

quantize_dynamic(preprocessed_name, quantize_name,
                 weight_type=QuantType.QUInt8)

res = compare_with(sess_full, imgs, quantize_name)
res["name"] = "dynamic"
experiments.append(res)
print(f"Discrepancies: mean={res['mean_diff']:.2f}, "
      f"mean_max={res['mean_max']:.2f}, "
      f"times {res['time_full']} -> {res['time_quant']}, "
      f"disagreement={res['disagree']:.2f}")
res

#######################################
# Conclusion
# ==========
#
# The static quantization (same quantized parameters for all observations)
# is not really working. The quantized
# model disagrees on almost all observations. Dynamic quantization
# (quantized parameters different for each observation)
# is a lot better but a lot slower too.

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
df = pandas.DataFrame(experiments).set_index("name")
df[["ratio"]].plot(ax=ax[0], title="Speedup\nlower better", kind="bar")
df[["mean_diff"]].plot(ax=ax[1], title="Average difference", kind="bar")
df[["disagree"]].plot(
    ax=ax[2], title="Proportion bast class is the same", kind="bar")

# plt.show()
