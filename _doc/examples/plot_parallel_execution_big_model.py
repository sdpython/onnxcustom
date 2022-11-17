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
This example uses the same models as in :ref:`l-plot-parallel-execution`.

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
    maxN, stepN, repN = 10, 1, 4
    big_model = True
elif small:
    model_name = "mobilenetv2-10.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/mobilenet/model")
    maxN, stepN, repN = 41, 2, 4
    big_model = False
else:
    model_name = "resnet18-v1-7.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/resnet/model")
    maxN, stepN, repN = 41, 2, 4
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

###########################################
# Pieces are roughly of the same size.
# Let's save them on disk.

piece_names = []
for i, piece in enumerate(pieces):
    name = f"piece-{os.path.splitext(model_name)[0]}-{i}.onnx"
    piece_names.append(name)
    with open(name, "wb") as f:
        f.write(piece.SerializeToString())

###########################################
# Discrepancies?
# ==============
#
# We need to make sure the split model is equivalent to
# the original one. Some data first.

sess_full = InferenceSession(model_name, providers=["CPUExecutionProvider"])

#############################
# inputs.

for i in sess_full.get_inputs():
    print(f"input {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    input_name = i.name
    input_shape = list(i.shape)
    if input_shape[0] in [None, "batch_size", "N"]:
        input_shape[0] = 1

#############################
# outputs.

output_name = None
for i in sess_full.get_outputs():
    print(f"output {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    if output_name is None:
        output_name = i.name

print(f"input_name={input_name!r}, output_name={output_name!r}")

#############################
# data

if model_name == "gpt2.onnx":
    imgs = [x["input_ids"].numpy()
            for x in encoded_tensors[:maxN]]
else:
    imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
            for i in range(maxN)]


###########################################
# The split model.

sess_split = [InferenceSession(name, providers=["CPUExecutionProvider"])
              for name in piece_names]
input_names = [sess.get_inputs()[0].name for sess in sess_split]

##########################################
# We are ready to compute the outputs from both models.

expected = sess_full.run(None, {input_name: imgs[0]})[0]

x = imgs[0]
for sess, name in zip(sess_split, input_names):
    feeds = {name: x}
    x = sess.run(None, feeds)[0]

diff = numpy.abs(expected - x).max()
print(f"Max difference: {diff}")

##########################################
# Everything works.
#
# Parallelization on GPU
# ======================
#
# First the implementation of a sequence.


def get_ort_device(sess):
    providers = sess.get_providers()
    if providers == ["CPUExecutionProvider"]:
        return C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    if providers[0] == "CUDAExecutionProvider":
        options = sess.get_provider_options()
        if len(options) == 0:
            return C_OrtDevice(C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)
        if "CUDAExecutionProvider" not in options:
            raise NotImplementedError(
                f"Unable to guess 'device_id' in {options}.")
        cuda = options["CUDAExecutionProvider"]
        if "device_id" not in cuda:
            raise NotImplementedError(
                f"Unable to guess 'device_id' in {options}.")
        device_id = int(cuda["device_id"])
        return C_OrtDevice(C_OrtDevice.cuda(), C_OrtDevice.default_memory(), device_id)
    raise NotImplementedError(
        f"Not able to guess the model device from {providers}.")


def sequence_ort_value(sesss, imgs):
    assert len(sesss) == 1
    sess = sesss[0]
    ort_device = get_ort_device(sess)
    res = []
    for img in imgs:
        ov = C_OrtValue.ortvalue_from_numpy(img, ort_device)
        out = sess._sess.run_with_ort_values(
            {input_name: ov}, [output_name], None)[0]
        res.append(out.numpy())
    return res


##################################
# And the parallel execution.

class MyThreadOrtValue(threading.Thread):

    def __init__(self, sess, n_imgs, next_thread=None, wait_time=1e-2):
        threading.Thread.__init__(self)
        self.sess = sess
        self.imgs = [None for i in range(n_imgs)]
        self.pos = 0
        self.q = []
        self.ort_device = ort_device
        self.wait_time = wait_time
        self.waiting_time = 0
        self.ort_device = get_ort_device(self.sess)
        self.next_thread = next_thread

    def append(self, img):
        if not isinstance(img, numpy.ndarray):
            raise TypeError(f"numpy array expected not {type(img)}.")
        if self.pos >= len(self.n_imgs):
            raise RuntimeError(
                f"Cannot append an image, {self.pos} were processed. "
                f"The thread should be finished.")
        self.imgs[self.pos] = img
        self.pos += 1

    def run(self):
        ort_device = self.ort_device
        sess = self.sess._sess

        while len(self.q) < len(self.imgs):

            # wait for an image
            while len(self.q) == self.pos:
                self.waiting_time += self.wait_time
                time.sleep(self.wait_time)

            ov = C_OrtValue.ortvalue_from_numpy(
                self.imgs[self.pos], ort_device)
            out = sess.run_with_ort_values(
                {input_name: ov}, [output_name], None)[0]
            cpu_res = out.numpy()
            q.append(cpu_res)

            # sent the result to the next part
            if self.next_thread is not None:
                self.next_thread.append(cpu_res)


def parallel_ort_value(sesss, imgs, wait_time=1e-2):
    n_parts = len(sesss)
    threads = []
    for i in range(len(sesss)):
        sess = sesss[-i - 1]
        next_thread == threads[-1] if i > 0 else None
        th = MyThreadOrtValue(sess, next_thread, wait_time=wait_time)
        threads.append(th)
    threads = list(reversed(threads))

    for img in imgs:
        threads[0].append(img)
    for t in threads:
        t.start()
    res = []
    for t in threads:
        t.join()
        res.extend(t.q)
    return res

###############################
# Functions
# =========
#
# Both functions `benchmark` and `make_plot` are adapted
# from example :ref:`l-plot-parallel-execution`.
# They produce the same figures. The main difference
# is the model used to compute the figures has to be
# deleted before running the other benchmark to free
# the GPU memory.


def benchmark(fcts, model_name, piece_names, imgs, stepN=1, repN=4):
    data = []
    Ns = [1] + list(range(nth, len(imgs), stepN * nth))
    ns_name = {}
    for name, build_fct, fct, index in fcts:
        ns_name[name] = []
        sesss = build_fct()
        for N in tqdm.tqdm(Ns):
            for i in range(repN):
                r = fct(sesss, imgs[:N])
                if i == 0:
                    # let's get rid of the first iteration sometimes
                    # used to initialize internal objects.
                    begin = time.perf_counter()
            end = (time.perf_counter() - begin) / (repN - 1)
            obs = {'n_imgs': len(imgs), 'maxN': maxN,
                   'stepN': stepN, 'repN': repN,
                   'batch_size': N, 'n_threads': len(sesss)}
            obs.update({f"n_imgs_{name}": len(r), f"time_{name}": end})
            ns_name[name].append(len(r))

        del sesss
        gc.collect()

        data.append(obs)

    names = list(ns_name)
    baseline = ns_name[names[0]]
    for name in names[1:]:
        if ns_name[name] != baseline:
            raise RuntimeError(
                f"Cannot compare experiments as it returns differents number of results, "
                f"ns_name={ns_name}, obs={obs}.")

    return pandas.DataFrame(data)


def make_plot(df, title):
    if df is None:
        return None
    if "n_threads" in df.columns:
        n_threads = list(set(df.n_threads))
        if len(n_threads) != 1:
            raise RuntimeError(f"n_threads={n_threads} must be unique.")
        index = "batch_size"
    else:
        n_threads = "?"
        index = "n_imgs_seq_cpu"
    kwargs = dict(title=f"{title}\nn_threads={n_threads}", logy=True)
    columns = [index] + [c for c in df.columns if c.startswith("time")]
    ax = df[columns].set_index(columns[0]).plot(**kwargs)
    ax.set_xlabel("batch size")
    ax.set_ylabel("seconds")
    return ax


###########################################
# Benchmark
# =========
#

###################################################
# Initialization
#

if n_gpus > 1:
    print("ORT // GPUs")
    if model_name == "gpt2.onnx":
        imgs = [x["input_ids"].numpy()
                for x in encoded_tensors[:maxN * len(sesss)]]
    else:
        imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
                for i in range(maxN * len(sesss))]

    df = benchmark(model_name=model_name, piece_names=piece_names,
                   imgs=imgs, stepN=stepN, repN=repN,
                   fcts=[('sequence', sequence_ort_value, 0),
                         ('parallel', parallel_ort_value, None)])
    df.reset_index(drop=False).to_csv("ort_gpus_piece.csv", index=False)

    del sesss[:]
    gc.collect()
else:
    print("No GPU is available but data should be like the following.")
    df = pandas.read_csv("data/ort_gpus_piece.csv").set_index("N")

df


####################################
# Plots.

ax = make_plot(df, f"Time per image / batch size\n{n_gpus} GPUs")
ax

####################################
# The parallelization on multiple GPUs did work.
# With a model `GPT2 <https://huggingface.co/gpt2?text=My+name+is+Mariama%2C+my+favorite>`_,
# it would give the following results.

data = pandas.read_csv("data/ort_gpus_piece_gpt2.csv")
df = pandas.DataFrame(data)
ax = make_plot(df, f"Time per image / batch size\n{n_gpus} GPUs - GPT2")
ax


# import matplotlib.pyplot as plt
# plt.show()
