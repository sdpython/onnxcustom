"""
.. _l-example-float8:

float 8
=======

Two papers have been published in 2022 to introduce floats
stored on a byte as opposed to float 32 stored on 4 bytes.
The float precision is much lower but the training precision
does not suffer too much.

`FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_
from NVIDIA introduces two types following
`IEEE specifciations <https://en.wikipedia.org/wiki/IEEE_754>`_.
First one is E4M3, 1 bit for the sign, 4 bits for the exponents and 3
bits for the mantissa. Second one is E5M2, 1 bit for the sign,
3 bits for the exponents and 2 for the mantissa. The first types
is mostly used for the coefficients, the second one for the gradient.

Second paper `8-bit Numerical Formats For Deep Neural Networks
<https://arxiv.org/pdf/2206.02915.pdf>`_ introduces
similar types. IEEE standard gives the same value
to `+0` (or integer 0) and `-0` (or integer 128).
They chose to give distinct float values to these two
numbers. The paper experiments different split between
exponent and mantissa and shows and E4M3 and E5M2 are
the best ones.

:math:`S` stands for the sign. :math:`10_2` describe a number base 2.

.. list-table:: Float8 types
   :widths: 10 10 10
   :header-rows: 1

   * - 
     - E4M3
     - E5M2
   * - Exponent bias
     - 7
     - 15
   * - Infinities
     -
     - :math:`S.11111.00_2`
   * - NaN
     - :math:`S.1111.111_2`
     - :math:`S.11111.\{01, 10, 11\}_2`
   * - Zeros
     - :math:`S.0000.000_2`
     - :math:`S.00000.00_2`
   * - Max
     - :math:`S.1111.110_2`
     - :math:`1.75 \times 2^{15}= 57344`
   * - Min
     - :math:`S.0000.001_2 = 2^{-9}`
     - :math:`S.00000.01_2 = 2^{-16}`


Let's denote the bit representation as :math:`S.b_6 b_5 b_4 b_3 b_2 b_1 b_0`.
The float value is defined by the following expressions:

.. list-table:: Float8 types values
   :widths: 10 10 10
   :header-rows: 1

   * - 
     - E4M3
     - E5M2
   * - exponent :math:`\neq` 0
     - :math:`(-1)^S 2^{\sum_{i=3}^6 b_i 2^{i-3} - 7} \sum_{i=0}^2 b_i 2^{i-2}`
     - :math:`(-1)^S 2^{\sum_{i=2}^6 b_i 2^{i-2} - 15} \sum_{i=0}^1 b_i 2^{i-1}`
   * - exponent :math:`=` 0
     - :math:`(-1)^S 2^{-6} \sum_{i=0}^2 b_i 2^{i-3}`
     - :math:`(-1)^S 2^{-14} \sum_{i=0}^1 b_i 2^{i-2}`

Cast from float 8 to
`float 16 <https://en.wikipedia.org/wiki/Half-precision_floating-point_format>`_ (or E5M10),
`bfloat16 <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_ (or E8M7),
`float32 <https://en.wikipedia.org/wiki/Single-precision_floating-point_format>`_ (or E8M23) is easier.
The cast is exact. The tricky part is to distinguish between exponent = 0 and :math:`neq 0`.

Cast to float 8 consists in finding the closest float 8
to the original float 32 value. It is usually done by shifting
and truncating. The tricky part is to handle rounding.

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
    display_fe4m3, display_fe5m2,
    fe4m3_to_float32, fe5m2_to_float32,
    float32_to_fe4m3, float32_to_fe5m2)

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

#######################################
# E5M2
# ++++
#
# List of possibles values:


values = [(fe5m2_to_float32(i), i, display_fe5m2(i)) for i in range(0, 256)]
values.sort()
values = [i[::-1] for i in values]

pprint.pprint(values)

######################################
# Round conversion.

for x in numpy.random.randn(10).astype(numpy.float32):
    f8 = float32_to_fe5m2(x)
    y = fe5m2_to_float32(f8)
    print(f"x={x}, f8={f8} or {display_fe5m2(f8)}, y={y}")
    f8_2 = float32_to_fe5m2(y)

###########################
# Bigger values.

for x in (numpy.random.rand(10) * 60000).astype(numpy.float32):
    f8 = float32_to_fe5m2(x)
    y = fe5m2_to_float32(f8)
    print(f"x={x}, f8={f8} or {display_fe5m2(f8)}, y={y}")
    f8_2 = float32_to_fe5m2(y)

#######################################
# Plot.

df = pandas.DataFrame(values)
df.columns = ["binary", "int", "float"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df["float"].plot(title="E5M2 values", ax=ax[0])
df["float"].plot(title="logarithmic scale", ax=ax[1], logy=True)
fig.savefig("fig.png")
