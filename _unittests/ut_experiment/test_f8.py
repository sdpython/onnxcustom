"""
@brief      test log(time=3s)
"""
import struct
import unittest
import numpy
from pyquickhelper.pycode import (
    ExtTestCase, skipif_travis, skipif_circleci, get_temp_folder)
from onnxcustom.experiment.f8 import (
    display_fe4m3,
    display_float16,
    display_float32,
    fe4m3_to_float32,
    float16_to_float32,
    float32_to_fe4m3,
    float32_to_float16,
    search_float32_into_fe4m3)


class TestF8(ExtTestCase):

    def test_fe4m3_to_float32_paper(self):
        self.assertEqual(fe4m3_to_float32(int("1111110", 2)), 448)
        self.assertEqual(fe4m3_to_float32(int("1000", 2)), 2 ** (-6))
        self.assertEqual(fe4m3_to_float32(int("1", 2)), 2 ** (-9))
        self.assertEqual(fe4m3_to_float32(int("111", 2)), 0.875 * 2 ** (-6))
        self.assertRaise(lambda: fe4m3_to_float32(256), ValueError)

    def test_display_float32(self):
        f = 45
        s = display_float32(45)
        self.assertEqual(s, "0.10000100.01101000000000000000000")
        s = display_fe4m3(45)
        self.assertEqual(s, "0.0101.101")
        s = display_float16(numpy.float16(45))
        self.assertEqual(s, "0.10100.0110100000")

    def test_float16_to_float32(self):
        fs = [numpy.float16(x) for x in [0, 1, numpy.nan, numpy.inf, -numpy.inf,
                                         10, 1000, 0.456, -0.456,
                                         0.25]]
        for f in fs:
            with self.subTest(f=f):
                x = float16_to_float32(f)
                if numpy.isnan(f):
                    self.assertTrue(numpy.isnan(x))
                    continue
                self.assertEqual(f.astype(numpy.float32), x)

    def test_float32_to_float16(self):
        fs = [numpy.float32(x) for x in [
            0, 1, numpy.nan, numpy.inf, -numpy.inf,
            10, 1000,
            0.25, -0.25,
            # 0.456, -0.456, 456.456, -456.456,  # failing values
        ]]
        for f in fs:
            with self.subTest(f=f):
                x = float32_to_float16(f)
                if numpy.isnan(f):
                    self.assertTrue(numpy.isnan(x))
                    continue
                self.assertEqual(f.astype(numpy.float16), x)

    def test_search_float32_into_fe4m3(self):
        values = [(fe4m3_to_float32(i), i) for i in range(0, 256)]
        values.sort()

        for value, expected in values:
            b = search_float32_into_fe4m3(value)
            ival = int.from_bytes(struct.pack("<f", value), "little")
            nf = float32_to_fe4m3(value)
            self.assertEqual(value, fe4m3_to_float32(b))

        for value in [1e-3, 1e-2, 1e-1, 0, 4, 5, 6, 7, 100, 200, 300]:
            with self.subTest(value=value):
                b = search_float32_into_fe4m3(value)
                nf = float32_to_fe4m3(value)
                self.assertEqual(b, nf)
            with self.subTest(value=-value):
                b = search_float32_into_fe4m3(-value)
                nf = float32_to_fe4m3(-value)
                self.assertEqual(b, nf)


if __name__ == "__main__":
    unittest.main(verbosity=2)
