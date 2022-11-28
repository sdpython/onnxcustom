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
import os
import pickle
import urllib.request
import threading
import time
import sys
import tqdm
import numpy
from numpy.testing import assert_allclose
import pandas
import onnx
import matplotlib.pyplot as plt
import torch.cuda
from onnxruntime import InferenceSession, get_all_providers
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from onnxcustom.utils.onnx_split import split_onnx
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


small = "custom" if "custom" in sys.argv else "big" not in sys.argv
if small == "custom":
    model_name = "gpt2.onnx"
    url_name = None
    maxN, stepN, repN = 10, 1, 4
    big_model = True
elif small:
    model_name = "mobilenetv2-10.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/mobilenet/model")
    maxN, stepN, repN = 81, 2, 4
    big_model = False
else:
    model_name = "resnet18-v1-7.onnx"
    url_name = ("https://github.com/onnx/models/raw/main/vision/"
                "classification/resnet/model")
    maxN, stepN, repN = 81, 2, 4
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

if model_name == "resnet18-v1-7.onnx":
    # best cutting point to parallelize on 2 GPUs
    cut_points = ["resnetv15_stage3_activation0"]
    n_parts = None
else:
    cut_points = None
    n_parts = max(n_gpus, 2)

pieces = split_onnx(model, n_parts=n_parts,
                    cut_points=cut_points, verbose=2)

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
    with open("encoded_tensors-gpt2.pkl", "rb") as f:
        [encoded_tensors, labels] = pickle.load(f)
    imgs = [x["input_ids"].numpy()
            for x in encoded_tensors[:maxN]]
else:
    imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)
            for i in range(maxN)]


###########################################
# The split model.

sess_split = []
for name in piece_names:
    try:
        sess_split.append(InferenceSession(
            name, providers=["CPUExecutionProvider"]))
    except InvalidArgument as e:
        raise RuntimeError(f"Part {name!r} cannot be loaded.") from e
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


def sequence_ort_value(sesss, imgs):
    assert len(sesss) == 1
    sess = sesss[0]
    ort_device = get_ort_device_from_session(sess)
    res = []
    for img in imgs:
        ov = C_OrtValue.ortvalue_from_numpy(img, ort_device)
        out = sess._sess.run_with_ort_values(
            {input_name: ov}, [output_name], None)[0]
        res.append(out.numpy())
    return res, {}, True


##################################
# And the parallel execution.

class MyThreadOrtValue(threading.Thread):

    def __init__(self, sess, batch_size, next_thread=None, wait_time=1e-4):
        threading.Thread.__init__(self)
        if batch_size <= 0:
            raise ValueError(f"batch_size={batch_size} must be positive.")
        self.sess = sess
        self.wait_time = wait_time
        self.ort_device = get_ort_device_from_session(self.sess)
        self.next_thread = next_thread
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.batch_size = batch_size

        # for the execution
        self.inputs = []
        self.outputs = []
        self.waiting_time0 = 0
        self.waiting_time = 0
        self.run_time = 0
        self.copy_time_1 = 0
        self.copy_time_2 = 0
        self.twait_time = 0
        self.total_time = 0

    def append(self, pos, img):
        if not isinstance(img, numpy.ndarray):
            raise TypeError(f"numpy array expected not {type(img)}.")
        if pos >= self.batch_size or pos < 0:
            raise RuntimeError(
                f"Cannot append an image, pos={pos} no in [0, {self.batch_size}[. "
                f"The thread should be finished.")
        self.inputs.append((pos, img))

    def run(self):
        ort_device = self.ort_device
        sess = self.sess._sess
        processed = 0

        while processed < self.batch_size:

            # wait for an image
            tw = time.perf_counter()

            while processed >= len(self.inputs):
                self.waiting_time += self.wait_time
                if len(self.inputs) == 0:
                    self.waiting_time0 += self.wait_time
                time.sleep(self.wait_time)

            pos, img = self.inputs[processed]
            t0 = time.perf_counter()
            ov = C_OrtValue.ortvalue_from_numpy(img, ort_device)
            t1 = time.perf_counter()
            out = sess.run_with_ort_values({self.input_name: ov},
                                           [self.output_name], None)[0]
            t2 = time.perf_counter()
            cpu_res = out.numpy()
            t3 = time.perf_counter()

            self.outputs.append((pos, cpu_res))

            # sent the result to the next part
            if self.next_thread is not None:
                self.next_thread.append(pos, cpu_res)
            self.inputs[processed] = None  # deletion
            processed += 1

            t4 = time.perf_counter()

            self.copy_time_1 += t1 - t0
            self.run_time += t2 - t1
            self.copy_time_2 += t3 - t2
            self.twait_time += t0 - tw
            self.total_time += t4 - tw


def parallel_ort_value(sesss, imgs, wait_time=1e-4):
    threads = []
    for i in range(len(sesss)):
        sess = sesss[-i - 1]
        next_thread = threads[-1] if i > 0 else None
        th = MyThreadOrtValue(
            sess, len(imgs), next_thread, wait_time=wait_time)
        threads.append(th)
    threads = list(reversed(threads))

    for i, img in enumerate(imgs):
        threads[0].append(i, img)
    for t in threads:
        t.start()
    res = []
    th = threads[-1]
    th.join()
    res.extend(th.outputs)
    indices = [r[0] for r in res]
    order = list(sorted(indices)) == indices
    res.sort()
    res = [r[1] for r in res]
    times = {"wait": [], "wait0": [], "copy1": [],
             "copy2": [], "run": [], "ttime": [], "wtime": []}
    for t in threads:
        times["wait"].append(t.waiting_time)
        times["wait0"].append(t.waiting_time0)
        times["copy1"].append(t.copy_time_1)
        times["copy2"].append(t.copy_time_2)
        times["run"].append(t.run_time)
        times["ttime"].append(t.total_time)
        times["wtime"].append(t.twait_time)
    return res, times, order

###############################
# Functions
# =========
#
# The benchmark runs one function on all batch sizes then
# deleted the model before going to the next function
# in order to free the GPU memory.


def benchmark(fcts, model_name, piece_names, imgs, stepN=1, repN=4):
    data = []
    Ns = list(range(1, len(imgs), stepN))
    ns_name = {}
    results = {}
    for name, build_fct, fct in fcts:
        ns_name[name] = []
        results[name] = []
        sesss = build_fct()
        fct(sesss, imgs[:2])
        for N in tqdm.tqdm(Ns):
            all_times = []
            begin = time.perf_counter()
            for i in range(repN):
                r, times, order = fct(sesss, imgs[:N])
                all_times.append(times)
            end = (time.perf_counter() - begin) / repN
            times = {}
            for key in all_times[0].keys():
                times[key] = sum(numpy.array(t[key]) for t in all_times) / repN
            obs = {'n_imgs': len(imgs), 'maxN': maxN,
                   'stepN': stepN, 'repN': repN,
                   'batch_size': N, 'n_threads': len(sesss),
                   'name': name}
            obs.update({"n_imgs": len(r), "time": end})
            obs['order'] = order
            if len(times) > 0:
                obs.update(
                    {f"wait0_{i}": t for i, t in enumerate(times["wait0"])})
                obs.update(
                    {f"wait_{i}": t for i, t in enumerate(times["wait"])})
                obs.update(
                    {f"copy1_{i}": t for i, t in enumerate(times["copy1"])})
                obs.update(
                    {f"copy2_{i}": t for i, t in enumerate(times["copy2"])})
                obs.update({f"run_{i}": t for i, t in enumerate(times["run"])})
                obs.update(
                    {f"ttime_{i}": t for i, t in enumerate(times["ttime"])})
                obs.update(
                    {f"wtime_{i}": t for i, t in enumerate(times["wtime"])})
            ns_name[name].append(len(r))
            results[name].append((r, obs))
            data.append(obs)

        del sesss
        gc.collect()

    # Let's maje sure again that the outputs are the same when the inference
    # is parallelized.
    names = list(ns_name)
    baseline = ns_name[names[0]]
    for name in names[1:]:
        if ns_name[name] != baseline:
            raise RuntimeError(
                f"Cannot compare experiments as it returns differents number of results, "
                f"ns_name={ns_name}, obs={obs}.")
    baseline = results[names[0]]
    for name in names[1:]:
        if len(results[name]) != len(baseline):
            raise RuntimeError("Cannot compare.")
        for i1, ((b, o1), (r, o2)) in enumerate(zip(baseline, results[name])):
            if len(b) != len(r):
                raise RuntimeError(
                    f"Cannot compare: len(b)={len(b)} != len(r)={len(r)}.")
            for i2, (x, y) in enumerate(zip(b, r)):
                try:
                    assert_allclose(x, y, atol=1e-3)
                except AssertionError as e:
                    raise AssertionError(
                        f"Issue with baseline={names[0]!r} and {name!r}, "
                        f"i1={i1}/{len(baseline)}, i2={i2}/{len(b)}\n"
                        f"o1={o1}\no2{o2}") from e

    return pandas.DataFrame(data)


def make_plot(df, title):
    if df is None:
        return None
    fig, ax = plt.subplots(3, 4, figsize=(12, 9), sharex=True)

    # perf
    a = ax[0, 0]
    perf = df.pivot(index="n_imgs", columns="name", values="time")
    num = perf["parallel"].copy()
    div = perf.index.values
    perf.plot(logy=True, ax=a)
    a.set_title("time(s)", fontsize="x-small")
    a.legend(fontsize="x-small")
    a.set_ylabel("seconds", fontsize="x-small")

    a = ax[0, 1]
    for c in perf.columns:
        perf[c] /= div
    perf.plot(ax=a)
    a.set_title("time(s) / batch_size", fontsize="x-small")
    a.legend(fontsize="x-small")
    a.set_ylim([0, None])
    a.set_ylabel("seconds", fontsize="x-small")
    a.set_xlabel("batch size", fontsize="x-small")

    a = ax[0, 2]
    perf["perf gain"] = (perf["sequence"] -
                         perf["parallel"]) / perf["sequence"]
    wcol = []
    wcol0 = []
    cs = []
    for i in range(0, 4):
        c = f"wait_{i}"
        if c not in df.columns:
            break
        wcol.append(c)
        wcol0.append(f"wait0_{i}")
        p = df.pivot(index="n_imgs", columns="name", values=c)
        perf[f"wait_{i}"] = p["parallel"].values / num
        cs.append(f"wait_{i}")
    n_parts = len(cs)

    perf["wait"] = perf[cs].sum(axis=1)
    perf[["perf gain", "wait"] + cs].plot(ax=a)
    a.set_title("gain / batch_size\n((baseline - parallel) / baseline",
                fontsize="x-small")
    a.legend(fontsize="x-small")
    a.set_ylim([0, None])
    a.set_ylabel("%", fontsize="x-small")
    a.set_xlabel("batch size", fontsize="x-small")

    # wait
    a = ax[1, 0]
    wait0 = df[["n_imgs"] + wcol0].set_index("n_imgs")
    wait0.plot(ax=a)
    a.set_title("Time waiting for the first image per thread",
                fontsize="x-small")
    a.legend(fontsize="x-small")
    a.set_ylabel("seconds", fontsize="x-small")

    a = ax[1, 1]
    wait = df[["n_imgs"] + wcol].set_index("n_imgs")
    wait.plot(ax=a)
    a.set_title("Total time waiting per thread", fontsize="x-small")
    a.legend(fontsize="x-small")
    a.set_ylabel("seconds", fontsize="x-small")

    a = ax[1, 2]
    wait = df[["n_imgs"] + wcol]
    div = wait["n_imgs"]
    wait = wait.set_index("n_imgs")
    for c in wait.columns:
        wait[c] /= div.values
    wait.plot(ax=a)
    a.set_title(
        "Total time waiting per thread\ndivided by batch size", fontsize="x-small")
    a.legend()
    a.set_ylim([0, None])
    a.set_ylabel("seconds", fontsize="x-small")
    a.set_xlabel("batch size", fontsize="x-small")
    a.set_xlabel("batch size", fontsize="x-small")

    # ttime
    a = ax[0, 3]
    ttimes = [c for c in df.columns if c.startswith('ttime_')]
    n_threads = len(ttimes)
    sub = df.loc[~df.run_0.isnull(), ["n_imgs", "time"] + ttimes].copy()
    for c in sub.columns[1:]:
        sub[c] /= sub["n_imgs"]
    sub.set_index("n_imgs").plot(ax=a, logy=True)
    a.set_title("Total time (parallel)\ndivided by batch size",
                fontsize="x-small")
    a.set_ylabel("seconds", fontsize="x-small")
    a.set_xlabel("batch size", fontsize="x-small")

    a = ax[1, 3]
    run = [c for c in df.columns if c.startswith('run_')]
    sub = df.loc[~df.run_0.isnull(), ["n_imgs", "time"] + run].copy()
    for c in sub.columns[2:]:
        sub[c] /= sub["time"]
    sub["time"] = 1
    sub.set_index("n_imgs").plot(ax=a)
    a.set_title(
        "Ratio running time per thread / total time\ndivided by batch size", fontsize="x-small")
    a.set_ylabel("seconds", fontsize="x-small")
    a.set_xlabel("batch size", fontsize="x-small")
    a.legend(fontsize="x-small")

    # other time
    pos = [2, 0]

    cols = ['wtime', 'copy1', 'run', 'copy2', 'ttime']
    for nth in range(n_threads):
        a = ax[tuple(pos)]
        cs = [f"{c}_{nth}" for c in cols]
        sub = df.loc[~df.run_0.isnull(), ["n_imgs", "time"] + cs].copy()
        for c in cs:
            sub[c] /= sub[f"ttime_{nth}"]
        sub.set_index("n_imgs")[cs[:-1]].plot.area(ax=a)
        a.set_title(f"Part {nth + 1}/{n_threads}", fontsize="x-small")
        a.set_xlabel("batch size", fontsize="x-small")
        a.legend(fontsize="x-small")

        pos[1] += 1
        if pos[1] >= ax.shape[1]:
            pos[1] = 0
            pos[0] += 1

    fig.suptitle(f"{title} - {n_parts} splits")
    fig.savefig(f"img-{n_parts}-splits-{title.replace(' ', '_')}.png", dpi=200)
    return ax


###########################################
# Benchmark
# =========
#

def build_sequence():
    return [InferenceSession(model_name, providers=["CUDAExecutionProvider",
                                                    "CPUExecutionProvider"])]


def build_parellel_pieces():
    sesss = []
    for i in range(len(piece_names)):
        print(f"Initialize device {i} with {piece_names[i]!r}")
        sesss.append(
            InferenceSession(
                piece_names[i],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                provider_options=[{"device_id": i}, {}]))
    return sesss


if n_gpus > 1:
    print("ORT // GPUs")

    df = benchmark(model_name=model_name, piece_names=piece_names,
                   imgs=imgs, stepN=stepN, repN=repN,
                   fcts=[('sequence', build_sequence, sequence_ort_value),
                         ('parallel', build_parellel_pieces, parallel_ort_value)])
    df.reset_index(drop=False).to_csv("ort_gpus_piece.csv", index=False)
    title = os.path.splitext(model_name)[0]
else:
    print("No GPU is available but data should be like the following.")
    df = pandas.read_csv("data/ort_gpus_piece.csv")
    title = "Saved mobilenet"

df


####################################
# Plots.

ax = make_plot(df, title)
ax

####################################
# Recorded results
# ================
#
# The parallelization on multiple GPUs did work.
# With a model resnet18.

data = pandas.read_csv("data/ort_gpus_piece_resnet18.csv")
df = pandas.DataFrame(data)
ax = make_plot(df, "Saved resnet 18")
ax


#######################################
# With `GPT2 <https://huggingface.co/gpt2?text=My+name+is+Mariama%2C+my+favorite>`_

if os.path.exists("data/ort_gpus_piece_gpt2.csv"):
    data = pandas.read_csv("data/ort_gpus_piece_gpt2.csv")
    df = pandas.DataFrame(data)
    ax = make_plot(df, "Saved GPT2")
else:
    print("No result yet.")
    ax = None
ax

# plt.show()
