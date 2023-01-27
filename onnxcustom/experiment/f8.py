# flake8: noqa: F401
"""
@file
@brief Helpers to manipulate float 8 number.
"""
import struct
import numpy


def display_float32(value, sign=1, exponent=8, mantissa=23):
    """
    Displays a float32 into b.

    :param value: value to display (float32)
    :param sign: number of bits for the sign
    :param exponent: number of bits for the exponent
    :param mantissa: number of bits for the mantissa
    :return: string
    """
    t = sign + exponent + mantissa
    ival = int.from_bytes(struct.pack("<f", numpy.float32(value)), "little")
    s = bin(ival)[2:]
    s = "0" * (t - len(s)) + s
    s1 = s[:sign]
    s2 = s[sign: sign + exponent]
    s3 = s[sign + exponent:]
    return ".".join([s1, s2, s3])


def display_float16(value, sign=1, exponent=5, mantissa=10):
    """
    Displays a float32 into b.

    :param value: value to display (float16)
    :param sign: number of bits for the sign
    :param exponent: number of bits for the exponent
    :param mantissa: number of bits for the mantissa
    :return: string
    """
    t = sign + exponent + mantissa
    ival = numpy.float16(value).view('H')
    s = bin(ival)[2:]
    s = "0" * (t - len(s)) + s
    s1 = s[:sign]
    s2 = s[sign: sign + exponent]
    s3 = s[sign + exponent:]
    return ".".join([s1, s2, s3])


def display_fe4m3(value, sign=1, exponent=4, mantissa=3):
    """
    Displays a float32 into b.

    :param value: value to display (int)
    :param sign: number of bits for the sign
    :param exponent: number of bits for the exponent
    :param mantissa: number of bits for the mantissa
    :return: string
    """
    t = sign + exponent + mantissa
    ival = value
    s = bin(ival)[2:]
    s = "0" * (t - len(s)) + s
    s1 = s[:sign]
    s2 = s[sign: sign + exponent]
    s3 = s[sign + exponent:]
    return ".".join([s1, s2, s3])


def fe4m3_to_float32_float(ival: int) -> float:
    """
    Casts a float 8 encoded as an integer into a float.

    :param ival: byte
    :return: float (float 32)
    """
    if ival < 0 or ival > 255:
        raise ValueError(f"{ival} is not a float8.")
    if ival == 255:
        return numpy.float32(numpy.inf)
    if ival == 127:
        return numpy.float32(-numpy.inf)
    if (ival & 0x7F) == 0:
        return numpy.float32(0)

    sign = ival & 0x80
    ival &= 0x7F
    expo = ival >> 3
    mant = ival & 0x07
    powe = expo & 0x0F
    if expo == 0:
        powe -= 6
        fraction = 0
    else:
        powe -= 7
        fraction = 1
    fval = float(mant / 8 + fraction) * 2.0**powe
    if sign:
        fval = -fval
    return numpy.float32(fval)


def fe4m3_to_float32(ival: int) -> float:
    """
    Casts a float 8 encoded as an integer into a float.

    :param ival: byte
    :return: float (float 32)
    """
    if ival < 0 or ival > 255:
        raise ValueError(f"{ival} is not a float8.")
    if ival == 255:
        return numpy.float32(numpy.inf)
    if ival == 127:
        return numpy.float32(-numpy.inf)

    expo = (ival & 0x78) >> 3
    mant = ival & 0x07
    sign = ival & 0x80
    res = sign << 24
    if expo == 0:
        if mant > 0:
            expo = 0x7F - 7
            if mant & 0x4 == 0:
                mant &= 0x3
                mant <<= 1
                expo -= 1
            if mant & 0x4 == 0:
                mant &= 0x3
                mant <<= 1
                expo -= 1
            if mant & 0x4 == 0:
                mant &= 0x3
                mant <<= 1
                expo -= 1
            res |= (mant & 0x3) << 21
            res |= expo << 23
    else:
        res |= mant << 20
        expo -= 0x7
        expo += 0x7F
        res |= expo << 23
    f = numpy.uint32(res).view(numpy.float32)
    return f


class CastFloat8:

    values_e4m3 = list(sorted((fe4m3_to_float32_float(i), i)
                       for i in range(0, 256)))

    @staticmethod
    def find_closest_value(value, sorted_values):
        """
        Search a value into a sorted array of values.

        :param value: float32 value to search
        :param sorted_values: list of tuple `[(float 32, byte)]`
        :return: byte

        The function searches into the first column the closest value and
        return the value on the second columns.
        """
        a = 0
        b = 256
        while a < b:
            m = (a + b) // 2
            th = sorted_values[m][0]
            if value == th:
                return sorted_values[m][1]
            if value < th:
                b = m
            elif a == m:
                break
            else:
                a = m
        # finds the closest one
        d1 = value - sorted_values[a][0]
        d2 = sorted_values[b][0] - value
        if d1 < d2:
            return sorted_values[a][1]
        return sorted_values[b][1]


def search_float32_into_fe4m3(value: float) -> int:
    """
    Casts a float 32 into a float E4M3.

    :param value: float
    :return: byte
    """
    f = numpy.float32(value)
    return CastFloat8.find_closest_value(f, CastFloat8.values_e4m3)


def float16_to_float32(x):
    """
    Converts a float16 into a float32.

    :param x: numpy.float16
    :return: numpy.float32


    See `32-bit to 16-bit Floating Point Conversion
    <https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion>`_.

    ::

        float half_to_float(const ushort x) {
            // IEEE-754 16-bit floating-point format (without infinity):
            // 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
            const uint e = (x&0x7C00)>>10; // exponent
            const uint m = (x&0x03FF)<<13; // mantissa
            // evil log2 bit hack to count leading zeros in denormalized format
            const uint v = as_uint((float)m)>>23;
            // sign : normalized : denormalized
            return as_float( (x&0x8000)<<16 |
                            (e!=0)*((e+112)<<23|m) |
                            ((e==0)&(m!=0))*((v-37)<<23|
                            ((m<<(150-v))&0x007FE000)));
        }
    """
    ival = numpy.float16(x).view('H')
    if ival == 0x7e00:
        return numpy.float32(numpy.nan)
    if ival == 0x7c00:
        return numpy.float32(numpy.inf)
    if ival == 0xfc00:
        return numpy.float32(-numpy.inf)
    e = (ival & 0x7C00) >> 10  # exponent
    m = (ival & 0x03FF) << 13  # mantissa
    # evil log2 bit hack to count leading zeros in denormalized format
    ret = (ival & 0x8000) << 16  # sign
    if e != 0:
        ret |= (e + 0x70) << 23  # exponent
        ret |= m  # mantissa
    elif m != 0:
        # sign : normalized : denormalized
        v = int(float(m)) >> 23
        ret |= ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))
    return numpy.int64(ret).astype(numpy.uint32).view(numpy.float32)


def float32_to_float16(x):
    """
    Converts a float16 into a float32.

    :param x: numpy.float32
    :return: numpy.float16

    See `32-bit to 16-bit Floating Point Conversion
    <https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion>`_.

    ::

        ushort float_to_half(const float x) {
            // IEEE-754 16-bit floating-point format (without infinity):
            // 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
            // round-to-nearest-even: add last bit after truncated mantissa
            const uint b = as_uint(x)+0x00001000;
            // exponent
            const uint e = (b&0x7F800000)>>23;
            // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
            const uint m = b&0x007FFFFF;
            // sign : normalized : denormalized : saturate
            return  (b&0x80000000)>>16 |
                    (e>112)*((((e-112)<<10)&0x7C00)|m>>13) |
                    ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) |
                    (e>143)*0x7FFF;
        }
    """
    b = int.from_bytes(struct.pack("<f", numpy.float32(x)), "little")
    if b == 0x7fc00000:
        return numpy.float16(numpy.nan)
    if b == 0x7f800000:
        return numpy.float16(numpy.inf)
    if b == 0xff800000:
        return numpy.float16(-numpy.inf)
    e = (b & 0x7F800000) >> 23  # exponent
    m = b & 0x007FFFFF  # mantissa
    ret = (b & 0x80000000) >> 16  # sign
    if e > 0x70:
        # normalized
        ret |= ((e - 0x70) << 10) & 0x7C00
        ret |= m >> 13
    if 0x65 < e < 0x71:
        # denormalize
        ret |= (((0x007FF000 + m) >> (125 - e)) + 1) >> 1
    if e > 0x8f:
        # saturate
        ret |= 0x7FFF

    return numpy.int64(ret).astype(numpy.uint16).view(numpy.float16)


def float32_to_fe4m3(x):
    """
    Converts a float32 into a float E4M3.

    :param x: numpy.float32
    :return: byte
    """
    b = int.from_bytes(struct.pack("<f", numpy.float32(x)), "little")
    if b == 0x7fc00000:
        return 0xff
    if b == 0x7f800000:
        return 0xff
    if b == 0xff800000:
        return 0x7f
    e = (b & 0x7F800000) >> 23  # exponent
    m = b & 0x007FFFFF  # mantissa
    ret = (b & 0x80000000) >> 24  # sign

    if e != 0:
        if e < 0x75:
            return 0
        if e < 0x78:
            d = 0x78 - e
            if 2 - d < 0:
                print(e, m, x, display_fe4m3(search_float32_into_fe4m3(x)), fe4m3_to_float32(
                    search_float32_into_fe4m3(x)), display_float32(x))
                print(d, m >> 20, bin(m >> 20))
            ret |= 1 << (2 - d)
            ret |= m >> (20 + d)
        elif e < 0x88:
            ex = (e - 0x78) << 3
            if ex == 0:
                ret |= 0x4
                ret |= m >> 21
            else:
                ret |= ex
                ret |= m >> 20
                if m & 0x80000:
                    # rounding
                    ret += 1
        else:
            ret |= 126  # 01111110
    elif m == 0:
        return 0
    return int(ret)
