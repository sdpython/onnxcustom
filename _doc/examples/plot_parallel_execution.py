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
should show significant result when used on different processors (CPU, GPU)
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
    SessionIOBinding, OrtDevice as C_OrtDevice)


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


small = True
if small:
    model_name = "mobilenetv2-10.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/mobilenet/model")
else:
    model_name = "resnet18-v1-7.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/resnet/model")
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
for i in sess1.get_outputs():
    print(f"output {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    output_name = i.name


rnd_img = numpy.random.rand(*input_shape).astype(numpy.float32)

res = sess1.run(None, {input_name: rnd_img})
print(f"output: type={res[0].dtype}, shape={res[0].shape}")

print(measure_time(lambda: sess1.run(None, {input_name: rnd_img}),
                   div_by_number=True, repeat=10, number=10))

#############################################
# Parallelization
# +++++++++++++++
#
# We define the number of threads as the number of cores divided by 2.
# This is a dummy value. It should let a core to handle the main program.

n_threads = min(8, multiprocessing.cpu_count() - 1)
print(f"n_threads={n_threads}")


imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
        for i in range(n_threads)]

sesss = [InferenceSession(model_name, providers=["CPUExecutionProvider"])
         for i in range(n_threads)]

################################
# Let's measure the time for a sequence of images.


def sequence(sesss, imgs, N=1):
    res = []
    for sess, img in zip(sesss, imgs):
        for i in range(N):
            res.append(sess.run(None, {input_name: img})[0])
    return res


print(measure_time(lambda: sequence(sesss, imgs),
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
    threads = [MyThread(sess, [img] * N)
               for sess, img in zip(sesss, imgs)]
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

print("ORT // CPU")
data = []
rep = 2
maxN = 21
for N in tqdm.tqdm(range(1, maxN, 2)):
    begin = time.perf_counter()
    for i in range(rep):
        res1 = sequence(sesss, imgs, N)
    end = (time.perf_counter() - begin) / rep
    obs = dict(N=N, n_imgs_seq=len(res1), time_seq=end)

    begin = time.perf_counter()
    for i in range(rep):
        res2 = parallel(sesss, imgs, N)
    end = (time.perf_counter() - begin) / rep
    obs.update(dict(n_imgs_par=len(res2), time_par=end))

    data.append(obs)

df = pandas.DataFrame(data)
df.reset_index(drop=False).to_csv("ort_cpu.csv", index=False)
df

#####################################
# Let's free the memory.

del sesss[:]
gc.collect()

##########################################
# Plots
# +++++


def make_plot(df, title):
    df["time_seq_img"] = df["time_seq"] / df["n_imgs_seq"]
    df["time_par_img"] = df["time_par"] / df["n_imgs_par"]

    ax = df[["n_imgs_seq", "time_seq_img", "time_par_img"]].set_index("n_imgs_seq").plot(
        title=title)
    ax.set_xlabel("batch size")
    ax.set_ylabel("s")
    return ax


make_plot(df, "Time per image / batch size")

#######################################
# As expected, it does not really improve. It is like parallezing using
# both strategies, per kernel and per image, both trying to access all
# the process cores.

###################################################
# Same with another API based on OrtValue
# +++++++++++++++++++++++++++++++++++++++
#
# See :epkg:`l-ortvalue-doc`.


class MyThreadBind(threading.Thread):

    def __init__(self, sess, imgs, ort_device):
        threading.Thread.__init__(self)
        self.sess = sess
        self.imgs = imgs
        self.q = []
        self.bind = SessionIOBinding(self.sess._sess)
        self.ort_device = ort_device

    def run(self):
        bind = self.bind
        ort_device = self.ort_device
        bind.bind_output(output_name, ort_device)
        sess = self.sess._sess
        q = self.q
        for img in self.imgs:
            bind.bind_input(input_name, ort_device,
                            img.dtype, img.shape,
                            img.__array_interface__['data'][0])
            sess.run_with_iobinding(bind, None)
            ortvalues = bind.get_outputs()
            q.append(ortvalues)


def parallel_bind(sesss, imgs, N=1):
    ort_device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    threads = [MyThreadBind(sess, [img] * N, ort_device)
               for sess, img in zip(sesss, imgs)]
    for t in threads:
        t.start()
    res = []
    for t in threads:
        t.join()
        res.extend(t.q)
    return res


print("ORT (bind) // CPU")
data = []
for N in tqdm.tqdm(range(1, maxN, 2)):
    begin = time.perf_counter()
    for i in range(rep):
        res1 = sequence(sesss, imgs, N)
    end = (time.perf_counter() - begin) / rep
    obs = dict(N=N, n_imgs_seq=len(res1), time_seq=end)

    begin = time.perf_counter()
    for i in range(rep):
        res2 = parallel_bind(sesss, imgs, N)
    end = (time.perf_counter() - begin) / rep
    obs.update(dict(n_imgs_par=len(res2), time_par=end))

    data.append(obs)

del sesss[:]
gc.collect()
df = pandas.DataFrame(data)
df.reset_index(drop=False).to_csv("ort_cpu_bind.csv", index=False)
df


############################
# Plots.

make_plot(df, "Time per image / batch size\nrun_with_iobinding")

########################################
# GPU
# ===
#
# Let's check first it is possible.

has_cuda = "CUDAExecutionProvider" in get_all_providers()
if not has_cuda:
    print(f"No CUDA provider was detected in {get_all_providers()}.")

#########################################
# Parallelization GPU + CPU
# +++++++++++++++++++++++++

if has_cuda:
    n_threads = 2
    sesss = [InferenceSession(model_name, providers=["CPUExecutionProvider"]),
             InferenceSession(model_name, providers=["CUDAExecutionProvider",
                                                     "CPUExecutionProvider"])]
    imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
            for i in range(n_threads)]

    print("ORT // CPU + GPU")
    data = []
    for N in tqdm.tqdm(range(1, maxN, 2)):
        begin = time.perf_counter()
        for i in range(rep):
            res1 = sequence(sesss, imgs, N)
        end = (time.perf_counter() - begin) / rep
        obs = dict(N=N, n_imgs_seq=len(res1), time_seq=end)

        begin = time.perf_counter()
        for i in range(rep):
            res2 = parallel_bind(sesss, imgs, N)
        end = (time.perf_counter() - begin) / rep
        obs.update(dict(n_imgs_par=len(res2), time_par=end))

        data.append(obs)

    del sesss[:]
    gc.collect()
    df = pandas.DataFrame(data)
    df.reset_index(drop=False).to_csv("ort_cpu_gpu.csv", index=False)
else:
    print("No GPU is available but it should show something like the following.")
    df = pandas.read_csv("data/ort_cpu_gpu.csv").set_index("N")

df

####################################
# Plots.

if has_cuda:
    ax = make_plot(df, "Time per image / batch size\nCPU + GPU")
else:
    ax = None

ax

####################################
# The parallelization on mulitple CPU + GPUs is not really
# working with this simple setting. GPU should take more
# images as it is faster.

#########################################
# Parallelization on multiple GPUs
# ++++++++++++++++++++++++++++++++

n_gpus = torch.cuda.device_count() if has_cuda else 0
if n_gpus == 0:
    print("No GPU or one GPU was detected.")
elif n_gpus == 1:
    print("1 GPU was detected.")
else:
    print(f"{n_gpus} GPUs were detected.")


if n_gpus > 1:
    n_threads = 2
    sesss = []
    for i in range(n_gpus):
        print(f"Initialize device {i}")
        sesss.append(
            InferenceSession(model_name, providers=["CUDAExecutionProvider",
                                                    "CPUExecutionProvider"],
                             provider_options=[{"device_id": i}, {}]))
    imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
            for i in range(n_threads)]

    print("ORT // GPUs")
    data = []
    for N in tqdm.tqdm(range(1, maxN, 2)):
        begin = time.perf_counter()
        for i in range(rep):
            res1 = sequence(sesss, imgs, N)
        end = (time.perf_counter() - begin) / rep
        obs = dict(N=N, n_imgs_seq=len(res1), time_seq=end)

        begin = time.perf_counter()
        for i in range(rep):
            res2 = parallel_bind(sesss, imgs, N)
        end = (time.perf_counter() - begin) / rep
        obs.update(dict(n_imgs_par=len(res2), time_par=end))

        data.append(obs)

    del sesss[:]
    gc.collect()
    df = pandas.DataFrame(data)
    df.reset_index(drop=False).to_csv("ort_gpus.csv", index=False)
else:
    print("No GPU is available but it should show something like the following.")
    df = pandas.read_csv("data/ort_gpus.csv").set_index("N")

df


####################################
# Plots.

if n_gpus > 1:
    ax = make_plot(df, "Time per image / batch size\nCPU + GPU")
else:
    ax = None

ax

####################################
# The parallelization on mulitple GPUs did work.

# import matplotlib.pyplot as plt
# plt.show()
