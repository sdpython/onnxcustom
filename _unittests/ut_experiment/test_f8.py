"""
@brief      test log(time=3s)
"""
import os
import struct
import unittest
import numpy
import pandas
from pyquickhelper.pycode import ExtTestCase
from onnxcustom.experiment.f8 import (
    display_fe4m3,
    display_float16,
    display_float32,
    fe4m3_to_float32,
    fe4m3_to_float32_float,
    float32_to_fe4m3,
    search_float32_into_fe4m3)


class TestF8(ExtTestCase):

    def test_fe4m3_to_float32_float_paper(self):
        self.assertEqual(fe4m3_to_float32_float(int("1111110", 2)), 448)
        self.assertEqual(fe4m3_to_float32_float(int("1000", 2)), 2 ** (-6))
        self.assertEqual(fe4m3_to_float32_float(int("1", 2)), 2 ** (-9))
        self.assertEqual(
            fe4m3_to_float32_float(int("111", 2)), 0.875 * 2 ** (-6))
        self.assertRaise(lambda: fe4m3_to_float32_float(256), ValueError)

    def test_fe4m3_to_float32_paper(self):
        self.assertEqual(fe4m3_to_float32(int("1111110", 2)), 448)
        self.assertEqual(fe4m3_to_float32(int("1000", 2)), 2 ** (-6))
        self.assertEqual(fe4m3_to_float32(int("1", 2)), 2 ** (-9))
        self.assertEqual(fe4m3_to_float32(int("111", 2)), 0.875 * 2 ** (-6))
        self.assertRaise(lambda: fe4m3_to_float32(256), ValueError)

    def test_fe4m3_to_float32_all(self):
        for i in range(0, 256):
            a = fe4m3_to_float32_float(i)
            b = fe4m3_to_float32(i)
            self.assertEqual(a, b)

    def test_display_float32(self):
        f = 45
        s = display_float32(f)
        self.assertEqual(s, "0.10000100.01101000000000000000000")
        s = display_fe4m3(f)
        self.assertEqual(s, "0.0101.101")
        s = display_float16(numpy.float16(f))
        self.assertEqual(s, "0.10100.0110100000")

    def test_search_float32_into_fe4m3_simple(self):
        values = [
            (0.001953125, 0.001953125),
            (416, 416),
            (-447.5, -448),
            (23.5, 24),
            (192.5, 192),
            (79.5, 80),
        ]
        for v, expected in values:
            with self.subTest(v=v, expected=expected):
                b = search_float32_into_fe4m3(v)
                got = fe4m3_to_float32_float(b)
                self.assertEqual(expected, got)
                b = float32_to_fe4m3(v)
                got = fe4m3_to_float32_float(b)
                self.assertEqual(expected, got)

    def test_search_float32_into_fe4m3(self):
        values = [(fe4m3_to_float32_float(i), i) for i in range(0, 256)]
        values.sort()

        obs = []
        values += [(1e-8, 0), (-1e-8, 0), (1e8, 448), (-1e-8, -448)]
        wrong = 0
        for value, _ in values:
            for add in [0, -0.4, -1e-4, 1e-4, 0.4]:
                v = value + add
                b = search_float32_into_fe4m3(v)
                nf = float32_to_fe4m3(v)
                if b != nf:
                    # signed, not signed zero?
                    if (nf & 0x7F) == 0 and (b & 0x7F) == 0:
                        continue
                    wrong += 1
                    obs.append(dict(
                        value=v,
                        bin_value=display_float32(v),
                        expected=b,
                        float_expected=fe4m3_to_float32_float(b),
                        bin_expected=display_fe4m3(b),
                        got=nf,
                        bin_got=display_fe4m3(nf),
                        float_got=fe4m3_to_float32_float(nf),
                        ok="" if b == nf else "WRONG",
                        true=value,
                        add=add,
                    ))
        if wrong > 0:
            output = os.path.join(os.path.dirname(__file__),
                                  "temp_search_float32_into_fe4m3.xlsx")
            pandas.DataFrame(obs).to_excel(output)
            raise AssertionError(f"{wrong} conversion are wrong.")

    def test_search_float32_into_fe4m3_equal(self):
        values = [(fe4m3_to_float32_float(i), i) for i in range(0, 256)]
        values.sort()

        for value, expected in values:
            with self.subTest(value=value, expected=expected, bin=display_float32(value)):
                b = search_float32_into_fe4m3(value)
                nf = float32_to_fe4m3(value)
                if value != 0:
                    self.assertEqual(expected, b)
                    self.assertEqual(expected, nf)
                else:
                    self.assertIn(b, (0, 128))
                    self.assertIn(nf, (0, 128))


if __name__ == "__main__":
    TestF8().test_search_float32_into_fe4m3_equal()
    unittest.main(verbosity=2)
