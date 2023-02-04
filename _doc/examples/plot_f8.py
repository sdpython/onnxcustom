"""
.. _l-example-float8:

float 8
=======

Precision is not that important when it comes to train
a deep neural network. That's what the paper
`FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_
shows. It introduces two new types encoded on one byte:

* E4M3: 1 bit for the sign, 4 for the exponent, 3 for the mantissa
* E5M2: 1 bit for the sign, 5 for the exponent, 2 for the mantissa


.. index:: discrepencies, float8, float, E4M3, E5M2

.. contents::
    :local:


E4M3
++++

List of possibles values:
"""
import pprint
import numpy
import matplotlib.pyplot as plt
import pandas
from onnxcustom.experiment.f8 import (
    display_fe4m3,
    fe4m3_to_float32,
    float32_to_fe4m3)

values = [(fe4m3_to_float32(i), i, display_fe4m3(i)) for i in range(0, 256)]
values.sort()
values = [i[::-1] for i in values]

pprint.pprint(values)

######################################
# Round conversion.

for x in numpy.random.randn(10).astype(numpy.float32):
    f8 = float32_to_fe4m3(x)
    y = fe4m3_to_float32(f8)
    print(f"x={x}, f8={f8} or {display_fe4m3(f8)}, y={y}")
    f8_2 = float32_to_fe4m3(y)

###########################
# Bigger values.

for x in (numpy.random.rand(10) * 500).astype(numpy.float32):
    f8 = float32_to_fe4m3(x)
    y = fe4m3_to_float32(f8)
    print(f"x={x}, f8={f8} or {display_fe4m3(f8)}, y={y}")
    f8_2 = float32_to_fe4m3(y)

#######################################
# Plot.

df = pandas.DataFrame(values)
df.columns = ["binary", "int", "float"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df["float"].plot(title="E4M3 values", ax=ax[0])
df["float"].plot(title="logarithmic scale", ax=ax[1], logy=True)
# fig.savefig("fig.png")
