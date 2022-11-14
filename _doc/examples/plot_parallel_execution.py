"""
.. _l-plot-parallel-execution:

===============================
Multithreading with onnxruntime
===============================

.. index:: thread, parallel

Python implements multithreading but it is not in practice due to the GIL
(see :epkg:`Le GIL`). However, if most of the parallelized code is not creating
python object, this option becomes more interesting than creating several processes
trying to exchange data through sockets. :epkg:`onnxruntime` falls into that category.
For a big model such as a deeplearning model, this might be interesting.
However, :epkg:`onnxruntime` already parallelize the computation of
every operator (Gemm, MatMul) using all the CPU it can get so this approach
should show significant results when used on different processors (CPU, GPU)
in parallel.

.. contents::
    :local:

A model
=======

Let's retrieve a not so big model.
"""
import gc
import multiprocessing
import os
import pickle
import urllib.request
import threading
import time
import tqdm
import numpy
import pandas
from cpyquickhelper.numbers import measure_time
import torch.cuda
from onnxruntime import InferenceSession, get_all_providers
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtDevice as C_OrtDevice, OrtValue as C_OrtValue)


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


small = "custom"
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

#############################################
# Measuring inference time when parallelizing on CPU
# ==================================================
#
# Sequence
# ++++++++
#
# Let's create a random image.

sess1 = InferenceSession(model_name, providers=["CPUExecutionProvider"])
for i in sess1.get_inputs():
    print(f"input {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    input_name = i.name
    input_shape = list(i.shape)
    if input_shape[0] in [None, "batch_size", "N"]:
        input_shape[0] = 1
output_name = None
for i in sess1.get_outputs():
    print(f"output {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    if output_name is None:
        output_name = i.name

print(f"input_name={input_name!r}, output_name={output_name!r}")

if model_name == "gpt2.onnx":
    with open("encoded_tensors-gpt2.pkl", "rb") as f:
        [encoded_tensors, labels] = pickle.load(f)
    rnd_img = encoded_tensors[0]["input_ids"].numpy()
else:
    rnd_img = numpy.random.rand(*input_shape).astype(numpy.float32)

results = sess1.run(None, {input_name: rnd_img})
print(f"output: type={results[0].dtype}, shape={results[0].shape}")

print(measure_time(lambda: sess1.run(None, {input_name: rnd_img}),
                   div_by_number=True, repeat=3, number=3))

#############################################
# Parallelization
# +++++++++++++++
#
# We define a number of threads lower than the number of cores.

n_threads = min(4, multiprocessing.cpu_count() - 1)
print(f"n_threads={n_threads}")


if model_name == "gpt2.onnx":
    imgs = [x["input_ids"].numpy() for x in encoded_tensors[:n_threads]]
else:
    imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
            for i in range(n_threads)]

sesss = [InferenceSession(model_name, providers=["CPUExecutionProvider"])
         for i in range(n_threads)]

################################
# Let's measure the time for a sequence of images.


def sequence(sess, imgs, N=1):
    res = []
    for i in range(N):
        for img in imgs:
            res.append(sess.run(None, {input_name: img})[0])
    return res


print(measure_time(lambda: sequence(sesss[0], imgs),
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


def parallel(sesss, imgs, N=1):
    if len(imgs) < N:
        raise RuntimeError(f"N={N} must be >= {len(imgs)}=len(imgs)")
    threads = [MyThread(sess, imgs[:N]) for sess in sesss]
    for t in threads:
        t.start()
    res = []
    for t in threads:
        t.join()
        res.extend(t.q)
    return res


print(measure_time(lambda: parallel(sesss, imgs),
                   div_by_number=True, repeat=2, number=2))


###################################
# It is worse for one image. It is expected as mentioned in the introduction.
# Let's check for different number of images to parallelize.

if not big_model:
    print("ORT // CPU")
    data = []
    for N in tqdm.tqdm(range(1, maxN, stepN)):
        for i in range(repN):
            res1 = sequence(sesss[0], imgs, N)
            if i == 0:
                # let's get rid of the first iteration sometimes
                # used to initialize internal objects.
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs = dict(N=N, n_imgs_seq=len(res1), time_seq=end)

        for i in range(repN):
            res2 = parallel(sesss, imgs, N)
            if i == 0:
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs.update(dict(n_imgs_par=len(res2), time_par=end))

        data.append(obs)
    df = pandas.DataFrame(data)
    df.reset_index(drop=False).to_csv("ort_cpu.csv", index=False)
else:
    print("ORT // CPU skipped for a big model.")
    df = None
df

##########################################
# Plots
# +++++


def make_plot(df, title):

    kwargs = dict(title=title, logy=True)
    if "time_seq" in df.columns:
        df["time_seq_img"] = df["time_seq"] / df["n_imgs_seq"]
        df["time_par_img"] = df["time_par"] / df["n_imgs_par"]
        columns = ["n_imgs_seq", "time_seq_img", "time_par_img"]
    else:
        df["time_seq_img_cpu"] = df["time_seq_cpu"] / df["n_imgs_seq_cpu"]
        df["time_seq_img_gpu"] = df["time_seq_gpu"] / df["n_imgs_seq_gpu"]
        df["time_par_img"] = df["time_par"] / df["n_imgs_par"]
        columns = ["n_imgs_seq_cpu", "time_seq_img_cpu", "time_seq_img_gpu", "time_par_img"]

    ax = df[columns].set_index(columns[0]).plot(**kwargs)
    ax.set_xlabel("batch size")
    ax.set_ylabel("s")
    return ax


make_plot(df, "Time per image / batch size") if df is not None else None

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

def sequence_ort_value(sess, imgs, N=1):
    ort_device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    res = []
    for i in range(N):
        for img in imgs:
            ov = C_OrtValue.ortvalue_from_numpy(img, ort_device)
            out = sess._sess.run_with_ort_values({input_name: ov}, [output_name], None)[0]
            res.append(out.numpy())
    return res


class MyThreadOrtValue(threading.Thread):

    def __init__(self, sess, imgs, ort_device):
        threading.Thread.__init__(self)
        self.sess = sess
        self.imgs = imgs
        self.q = []
        self.ort_device = ort_device

    def run(self):
        ort_device = self.ort_device
        sess = self.sess._sess
        q = self.q
        for img in self.imgs:
            ov = C_OrtValue.ortvalue_from_numpy(img, ort_device)
            out = sess.run_with_ort_values({input_name: ov}, [output_name], None)[0]
            q.append(out.numpy())


def parallel_ort_value(sess, imgs, N=1):
    if len(imgs) < N:
        raise RuntimeError(f"N={N} must be >= {len(imgs)}=len(imgs)")
    ort_device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    threads = [MyThreadOrtValue(sess, imgs[:N], ort_device) for sess in sesss]
    for t in threads:
        t.start()
    res = []
    for t in threads:
        t.join()
        res.extend(t.q)
    return res


if not big_model:
    print("ORT (OrtValue) // CPU")
    data = []
    for N in tqdm.tqdm(range(1, maxN, stepN)):
        for i in range(repN):
            res1 = sequence_ort_value(sesss[0], imgs, N)
            if i == 0:
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs = dict(N=N, n_imgs_seq=len(res1), time_seq=end)

        for i in range(repN):
            res2 = parallel_ort_value(sesss, imgs, N)
            if i == 0:
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs.update(dict(n_imgs_par=len(res2), time_par=end))

        data.append(obs)

    df = pandas.DataFrame(data)
    df.reset_index(drop=False).to_csv("ort_cpu_ortvalue.csv", index=False)
else:
    df = None
    print("ORT (OrtValue) // CPU skipped for a long model")
df

#####################################
# Let's free the memory.

del sesss[:]
gc.collect()

############################
# Plots.

make_plot(df, "Time per image / batch size\nrun_with_ort_values") if df is not None else None

########################################
# It leads to the same conclusion. It is no use to parallelize
# on CPU as onnxruntime is already doing that per kernel.


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
    n_threads = 2
    repN = 4
    sesss = [InferenceSession(model_name, providers=["CPUExecutionProvider"]),
             InferenceSession(model_name, providers=["CUDAExecutionProvider",
                                                     "CPUExecutionProvider"])]
    if model_name == "gpt2.onnx":
        imgs = [x["input_ids"].numpy() for x in encoded_tensors[:maxN]]
    else:
        imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
                for i in range(maxN)]

    print("ORT // CPU + GPU")
    data = []
    for N in tqdm.tqdm(range(1, maxN, stepN)):
        for i in range(repN):
            res1 = sequence_ort_value(sesss[0], imgs, N)
            if i == 0:
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs = dict(N=N, n_imgs_seq_cpu=len(res1), time_seq_cpu=end)

        for i in range(repN):
            res2 = sequence_ort_value(sesss[1], imgs, N)
            if i == 0:
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs.update(dict(n_imgs_seq_gpu=len(res2), time_seq_gpu=end))

        for i in range(repN):
            res3 = parallel_ort_value(sesss, imgs, N)
            if i == 0:
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs.update(dict(n_imgs_par=len(res3), time_par=end))

        data.append(obs)

    del sesss[:]
    gc.collect()
    df = pandas.DataFrame(data)
    df.reset_index(drop=False).to_csv("ort_cpu_gpu.csv", index=False)
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
    n_threads = 2
    sesss = []
    for i in range(n_gpus):
        print(f"Initialize device {i}")
        sesss.append(
            InferenceSession(model_name, providers=["CUDAExecutionProvider",
                                                    "CPUExecutionProvider"],
                             provider_options=[{"device_id": i}, {}]))
    if model_name == "gpt2.onnx":
        imgs = [x["input_ids"].numpy() for x in encoded_tensors[:maxN]]
    else:
        imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
                for i in range(maxN)]

    print("ORT // GPUs")
    data = []
    for N in tqdm.tqdm(range(1, maxN, stepN)):
        for i in range(repN):
            res1 = sequence_ort_value(sess1, imgs, N)
            if i == 0:
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs = dict(N=N, n_imgs_seq_cpu=len(res1), time_seq_cpu=end)

        for i in range(repN):
            res2 = sequence_ort_value(sesss[0], imgs, N)
            if i == 0:
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs.update(dict(n_imgs_seq_gpu=len(res2), time_seq_gpu=end))

        for i in range(repN):
            res3 = parallel_ort_value(sesss, imgs, N)
            if i == 0:
                begin = time.perf_counter()
        end = (time.perf_counter() - begin) / (repN - 1)
        obs.update(dict(n_imgs_par=len(res3), time_par=end))

        data.append(obs)

    del sesss[:]
    gc.collect()
    df = pandas.DataFrame(data)
    df.reset_index(drop=False).to_csv("ort_gpus.csv", index=False)
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

# import matplotlib.pyplot as plt
# plt.show()
