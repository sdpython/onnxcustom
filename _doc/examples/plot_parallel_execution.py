"""
.. _l-plot-parallel-execution:

===============================
Multithreading with onnxruntime
===============================

.. index:: thread, parallel, onnxruntime

Python implements multithreading but it is not working in practice due to the GIL
(see :epkg:`Le GIL`). However, if most of the parallelized code is not creating
python object, this option becomes more interesting than creating several processes
trying to exchange data through sockets. :epkg:`onnxruntime` falls into that category.
For a big model such as a deeplearning model, this might be interesting.
:epkg:`onnxruntime` already parallelizes the computation of
every operator (Gemm, MatMul) using all the CPU it can get.
To use that approach to get significant results, it needs
to be used on different processors (CPU, GPU) in parallel.
That's what this example shows.

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
from onnxcustom.utils.benchmark import measure_time
import torch.cuda
from onnxruntime import InferenceSession, get_all_providers
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtDevice as C_OrtDevice, OrtValue as C_OrtValue)
from onnxcustom.utils.onnxruntime_helper import get_ort_device_from_session


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
    maxN, stepN, repN = 5, 1, 4
    big_model = True
elif small:
    model_name = "mobilenetv2-10.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/mobilenet/model")
    maxN, stepN, repN = 21, 2, 4
    big_model = False
else:
    model_name = "resnet18-v1-7.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/resnet/model")
    maxN, stepN, repN = 21, 2, 4
    big_model = False

if url_name is not None:
    url_name += "/" + model_name
    download_file(url_name, model_name, 100000)


#############################################
# Measuring inference time when parallelizing on CPU
# ==================================================
#
# Sequence
# ++++++++
#
# Let's first dig into the model to retrieve the input and output
# names as well as their shapes.

sess1 = InferenceSession(model_name, providers=["CPUExecutionProvider"])

#############################
# inputs.

for i in sess1.get_inputs():
    print(f"input {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    input_name = i.name
    input_shape = list(i.shape)
    if input_shape[0] in [None, "batch_size", "N"]:
        input_shape[0] = 1

#############################
# outputs.

output_name = None
for i in sess1.get_outputs():
    print(f"output {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    if output_name is None:
        output_name = i.name

print(f"input_name={input_name!r}, output_name={output_name!r}")

##########################################
# Let's take some random inputs.

if model_name == "gpt2.onnx":
    with open("encoded_tensors-gpt2.pkl", "rb") as f:
        [encoded_tensors, labels] = pickle.load(f)
    rnd_img = encoded_tensors[0]["input_ids"].numpy()
else:
    rnd_img = numpy.random.rand(*input_shape).astype(numpy.float32)

###################################
# And measure the processing time.

results = sess1.run(None, {input_name: rnd_img})

print(f"output: type={results[0].dtype}, shape={results[0].shape}")

pprint(measure_time(lambda: sess1.run(None, {input_name: rnd_img}),
                    div_by_number=True, repeat=3, number=3))

#############################################
# Parallelization
# +++++++++++++++
#
# We define a number of threads lower than the number of cores.

n_threads = min(4, multiprocessing.cpu_count() - 1)
print(f"n_threads={n_threads}")


if model_name == "gpt2.onnx":
    imgs = [x["input_ids"].numpy() for x in encoded_tensors[:maxN * n_threads]]
else:
    imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
            for i in range(maxN * n_threads)]

##############################################
# Let's create an object `InferenceSession` for every thread
# assuming the memory can hold that many objects.

sesss = [InferenceSession(model_name, providers=["CPUExecutionProvider"])
         for i in range(n_threads)]

################################
# Let's measure the time for a sequence of images.


def sequence(sess, imgs):
    # A simple function going through all images.
    res = []
    for img in imgs:
        res.append(sess.run(None, {input_name: img})[0])
    return res


pprint(measure_time(lambda: sequence(sesss[0], imgs[:n_threads]),
                    div_by_number=True, repeat=2, number=2))

#################################
# And then with multithreading.


class MyThread(threading.Thread):

    def __init__(self, sess, imgs):
        threading.Thread.__init__(self)
        self.sess = sess
        self.imgs = imgs
        self.q = []

    def run(self):
        for img in self.imgs:
            r = self.sess.run(None, {input_name: img})[0]
            self.q.append(r)


def parallel(sesss, imgs):
    # creation of the threads
    n_threads = len(sesss)
    threads = [MyThread(sess, imgs[i::n_threads])
               for i, sess in enumerate(sesss)]
    # start the threads
    for t in threads:
        t.start()
    # wait for each of them and stores the results
    res = []
    for t in threads:
        t.join()
        res.extend(t.q)
    return res


pprint(measure_time(lambda: parallel(sesss, imgs[:n_threads]),
                    div_by_number=True, repeat=2, number=2))


###################################
# It is worse. It is expected as this code tries to parallelize
# the execution of onnxruntime which is also trying to
# parallelize the execution of every matrix multiplication,
# every tensor operators. It is like using two conflicting strategies
# to parallize.
# Let's check for a different number of images to parallelize.

def benchmark(fcts, sesss, imgs, stepN=1, repN=4):
    data = []
    nth = len(sesss)
    Ns = [1] + list(range(nth, len(imgs), stepN * nth))
    for N in tqdm.tqdm(Ns):
        obs = {'n_imgs': len(imgs), 'maxN': maxN,
               'stepN': stepN, 'repN': repN,
               'batch_size': N, 'n_threads': len(sesss)}
        ns = []
        for name, fct, index in fcts:
            for i in range(repN):
                if index is None:
                    r = fct(sesss, imgs[:N])
                else:
                    r = fct(sesss[index], imgs[:N])
                if i == 0:
                    # let's get rid of the first iteration sometimes
                    # used to initialize internal objects.
                    begin = time.perf_counter()
            end = (time.perf_counter() - begin) / (repN - 1)
            obs.update({f"n_imgs_{name}": len(r), f"time_{name}": end})
            ns.append(len(r))

        if len(set(ns)) != 1:
            raise RuntimeError(
                f"Cannot compare experiments as it returns differents number of "
                f"results ns={ns}, obs={obs}.")
        data.append(obs)

    return pandas.DataFrame(data)


if not big_model:
    print(f"ORT // CPU, n_threads={len(sesss)}")
    df = benchmark(sesss=sesss, imgs=imgs, stepN=stepN, repN=repN,
                   fcts=[('sequence', sequence, 0),
                         ('parallel', parallel, None)])
    df.reset_index(drop=False).to_csv("ort_cpu.csv", index=False)
else:
    print("ORT // CPU skipped for a big model.")
    df = None
df

##########################################
# Plots
# +++++


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


make_plot(df, "Time per image / batch size")

#######################################
# As expected, it does not improve. It is like parallezing using
# two strategies, per kernel and per image, both trying to access all
# the process cores at the same time. The time spent to synchronize
# is significant.

###################################################
# Same with another API based on OrtValue
# +++++++++++++++++++++++++++++++++++++++
#
# See :epkg:`l-ortvalue-doc`.


def sequence_ort_value(sess, imgs):
    ort_device = get_ort_device_from_session(sess)
    res = []
    for img in imgs:
        ov = C_OrtValue.ortvalue_from_numpy(img, ort_device)
        out = sess._sess.run_with_ort_values(
            {input_name: ov}, [output_name], None)[0]
        res.append(out.numpy())
    return res


class MyThreadOrtValue(threading.Thread):

    def __init__(self, sess, imgs):
        threading.Thread.__init__(self)
        self.sess = sess
        self.imgs = imgs
        self.q = []
        self.ort_device = get_ort_device_from_session(self.sess)

    def run(self):
        ort_device = self.ort_device
        sess = self.sess._sess
        q = self.q
        for img in self.imgs:
            ov = C_OrtValue.ortvalue_from_numpy(img, ort_device)
            out = sess.run_with_ort_values(
                {input_name: ov}, [output_name], None)[0]
            q.append(out.numpy())


def parallel_ort_value(sess, imgs):
    n_threads = len(sesss)
    threads = [MyThreadOrtValue(sess, imgs[i::n_threads])
               for i, sess in enumerate(sesss)]
    for t in threads:
        t.start()
    res = []
    for t in threads:
        t.join()
        res.extend(t.q)
    return res


if not big_model:
    print(f"ORT // CPU (OrtValue), n_threads={len(sesss)}")
    df = benchmark(sesss=sesss, imgs=imgs, stepN=stepN, repN=repN,
                   fcts=[('sequence', sequence_ort_value, 0),
                         ('parallel', parallel_ort_value, None)])
    df.reset_index(drop=False).to_csv("ort_cpu_ortvalue.csv", index=False)
else:
    print("ORT // CPU (OrtValue) skipped for a big model.")
    df = None
df

############################
# Plots.

make_plot(df, "Time per image / batch size\nrun_with_ort_values")

#####################################
# It leads to the same conclusion. It is no use to parallelize
# on CPU as onnxruntime is already doing that per kernel.
# Let's free the memory to make some space
# for other experiments.

del sesss[:]
gc.collect()

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


#########################################
# Parallelization GPU + CPU
# +++++++++++++++++++++++++

if has_cuda and n_gpus > 0:
    print("ORT // CPU + GPU")
    repN = 4
    sesss = [InferenceSession(model_name, providers=["CPUExecutionProvider"]),
             InferenceSession(model_name, providers=["CUDAExecutionProvider",
                                                     "CPUExecutionProvider"])]
    if model_name == "gpt2.onnx":
        imgs = [x["input_ids"].numpy()
                for x in encoded_tensors[:maxN * len(sesss)]]
    else:
        imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
                for i in range(maxN * len(sesss))]

    df = benchmark(sesss=sesss, imgs=imgs, stepN=stepN, repN=repN,
                   fcts=[('seq_cpu', sequence_ort_value, 0),
                         ('seq_gpu', sequence_ort_value, 1),
                         ('parallel', parallel_ort_value, None)])
    df.reset_index(drop=False).to_csv("ort_cpu_gpu.csv", index=False)

    del sesss[:]
    gc.collect()
else:
    print("No GPU is available but data should be like the following.")
    df = pandas.read_csv("data/ort_cpu_gpu.csv").set_index("N")

df

####################################
# Plots.

ax = make_plot(df, "Time per image / batch size\nCPU + GPU")
ax

####################################
# The parallelization on mulitple CPU + GPUs is working, it is faster than CPU
# but it is still slower than using a single GPU in that case.

#########################################
# Parallelization on multiple GPUs
# ++++++++++++++++++++++++++++++++
#
# This is the only case for which it should work as every GPU is indenpendent.

if n_gpus > 1:
    print("ORT // GPUs")
    sesss = []
    for i in range(n_gpus):
        print(f"Initialize device {i}")
        sesss.append(
            InferenceSession(model_name, providers=["CUDAExecutionProvider",
                                                    "CPUExecutionProvider"],
                             provider_options=[{"device_id": i}, {}]))
    if model_name == "gpt2.onnx":
        imgs = [x["input_ids"].numpy()
                for x in encoded_tensors[:maxN * len(sesss)]]
    else:
        imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
                for i in range(maxN * len(sesss))]

    df = benchmark(sesss=sesss, imgs=imgs, stepN=stepN, repN=repN,
                   fcts=[('sequence', sequence_ort_value, 0),
                         ('parallel', parallel_ort_value, None)])
    df.reset_index(drop=False).to_csv("ort_gpus.csv", index=False)

    del sesss[:]
    gc.collect()
else:
    print("No GPU is available but data should be like the following.")
    df = pandas.read_csv("data/ort_gpus.csv").set_index("N")

df


####################################
# Plots.

ax = make_plot(df, f"Time per image / batch size\n{n_gpus} GPUs")
ax

####################################
# The parallelization on multiple GPUs did work.
# With a model `GPT2 <https://huggingface.co/gpt2?text=My+name+is+Mariama%2C+my+favorite>`_,
# it would give the following results.

data = pandas.read_csv("data/ort_gpus_gpt2.csv")
df = pandas.DataFrame(data)
ax = make_plot(df, f"Time per image / batch size\n{n_gpus} GPUs - GPT2")
ax


# import matplotlib.pyplot as plt
# plt.show()
