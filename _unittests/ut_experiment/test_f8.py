"""
@brief      test log(time=3s)
"""
import os
import unittest
import numpy
import pandas
from pyquickhelper.pycode import ExtTestCase
from onnxcustom.experiment.f8 import (
    CastFloat8,
    display_fe4m3,
    display_fe5m2,
    display_float16,
    display_float32,
    fe4m3_to_float32,
    fe5m2_to_float32,
    fe4m3_to_float32_float,
    fe5m2_to_float32_float,
    float32_to_fe4m3,
    float32_to_fe5m2,
    search_float32_into_fe4m3,
    search_float32_into_fe5m2)


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

    def test_fe5m2_to_float32_float_paper(self):
        self.assertEqual(fe5m2_to_float32_float(int("1111011", 2)), 57344)
        self.assertEqual(fe5m2_to_float32_float(int("100", 2)), 2 ** (-14))
        self.assertEqual(fe5m2_to_float32_float(
            int("11", 2)), 0.75 * 2 ** (-14))
        self.assertEqual(fe5m2_to_float32_float(int("1", 2)), 2 ** (-16))
        self.assertRaise(lambda: fe5m2_to_float32_float(256), ValueError)
        self.assertTrue(numpy.isnan(fe5m2_to_float32_float(int("1111101", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32_float(int("1111110", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32_float(int("1111111", 2))))
        self.assertTrue(numpy.isnan(
            fe5m2_to_float32_float(int("11111101", 2))))
        self.assertTrue(numpy.isnan(
            fe5m2_to_float32_float(int("11111110", 2))))
        self.assertTrue(numpy.isnan(
            fe5m2_to_float32_float(int("11111111", 2))))
        self.assertEqual(fe5m2_to_float32_float(int("1111100", 2)), numpy.inf)
        self.assertEqual(fe5m2_to_float32_float(
            int("11111100", 2)), -numpy.inf)

    def test_fe5m2_to_float32_paper(self):
        self.assertEqual(fe5m2_to_float32(int("1111011", 2)), 57344)
        self.assertEqual(fe5m2_to_float32(int("100", 2)), 2 ** (-14))
        self.assertEqual(fe5m2_to_float32(
            int("11", 2)), 0.75 * 2 ** (-14))
        self.assertEqual(fe5m2_to_float32(int("1", 2)), 2 ** (-16))
        self.assertRaise(lambda: fe5m2_to_float32(256), ValueError)
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("1111101", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("1111110", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("1111111", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("11111101", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("11111110", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("11111111", 2))))
        self.assertEqual(fe5m2_to_float32(int("1111100", 2)), numpy.inf)
        self.assertEqual(fe5m2_to_float32(int("11111100", 2)), -numpy.inf)

    def test_fe4m3_to_float32_all(self):
        for i in range(0, 256):
            a = fe4m3_to_float32_float(i)
            b = fe4m3_to_float32(i)
            if numpy.isnan(a):
                self.assertTrue(numpy.isnan(b))
                continue
            self.assertEqual(a, b)

    def test_display_float(self):
        f = 45
        s = display_float32(f)
        self.assertEqual(s, "0.10000100.01101000000000000000000")
        s = display_fe4m3(f)
        self.assertEqual(s, "0.0101.101")
        s = display_fe5m2(f)
        self.assertEqual(s, "0.01011.01")
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

    def test_search_float32_into_fe5m2_simple(self):
        values = [
            (0.0017089844, 0.0017089844),
            (20480, 20480),
            (20480.5, 20480),
            (14.5, 14),
            (-3584.5, -3584),
        ]
        for v, expected in values:
            with self.subTest(v=v, expected=expected):
                b = search_float32_into_fe5m2(v)
                got = fe5m2_to_float32_float(b)
                self.assertLess(abs(expected - got), 1e-5)
                b = float32_to_fe5m2(v)
                got = fe5m2_to_float32_float(b)
                self.assertLess(abs(expected - got), 1e-5)

    def test_search_float32_into_fe4m3_equal(self):
        values = [(fe4m3_to_float32_float(i), i) for i in range(0, 256)]
        values.sort()

        for value, expected in values:
            with self.subTest(value=value, expected=expected, bin=display_float32(value)):
                b = search_float32_into_fe4m3(value)
                nf = float32_to_fe4m3(value)
                if expected in (127, 255):
                    self.assertIn(b, (127, 255))
                    self.assertIn(nf, (127, 255))
                elif value != 0:
                    self.assertEqual(expected, b)
                    self.assertEqual(expected, nf)
                else:
                    self.assertIn(b, (0, 128))
                    self.assertIn(nf, (0, 128))

    def test_search_float32_into_fe5m2_equal(self):
        values = [(fe5m2_to_float32_float(i), i) for i in range(0, 256)]
        values.sort()

        for value, expected in values:
            with self.subTest(value=value, expected=expected, bin=display_float32(value)):
                b = search_float32_into_fe5m2(value)
                nf = float32_to_fe5m2(value)
                if expected in {253, 254, 255, 125, 126, 127}:  # nan
                    self.assertIn(b, {253, 254, 255, 125, 126, 127})
                    self.assertIn(nf, {253, 254, 255, 125, 126, 127})
                elif value != 0:
                    self.assertEqual(expected, b)
                    self.assertEqual(expected, nf)
                else:
                    self.assertIn(b, (0, 128))
                    self.assertIn(nf, (0, 128))

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

    def test_search_float32_into_fe5m2(self):
        values = [(fe5m2_to_float32_float(i), i) for i in range(0, 256)]
        values.sort()

        obs = []
        values += [(1e-8, 0), (-1e-8, 0), (1e8, 448), (-1e-8, -448)]
        wrong = 0
        for value, _ in values:
            for add in [0, -0.4, -1e-4, 1e-4, 0.4]:
                v = value + add
                b = search_float32_into_fe5m2(v)
                nf = float32_to_fe5m2(v)
                if b != nf:
                    # signed, not signed zero?
                    if (nf & 0x7F) == 0 and (b & 0x7F) == 0:
                        continue
                    wrong += 1
                    obs.append(dict(
                        value=v,
                        bin_value=display_float32(v),
                        expected=b,
                        float_expected=fe5m2_to_float32_float(b),
                        bin_expected=display_fe5m2(b),
                        got=nf,
                        bin_got=display_fe5m2(nf),
                        float_got=fe5m2_to_float32_float(nf),
                        ok="" if b == nf else "WRONG",
                        true=value,
                        add=add,
                    ))
        if wrong > 0:
            output = os.path.join(os.path.dirname(__file__),
                                  "temp_search_float32_into_fe5m2.xlsx")
            pandas.DataFrame(obs).to_excel(output)
            raise AssertionError(f"{wrong} conversion are wrong.")

    def test_inf_nan(self):
        np_fp32 = numpy.array([
            "0.47892547", "0.48033667", "0.49968487", "0.81910545",
            "0.47031248", "0.816468", "0.21087195", "0.7229038",
            "NaN", "INF", "+INF", "-INF"], dtype=numpy.float32)
        v_fe4m3_to_float32 = numpy.vectorize(fe4m3_to_float32)
        v_float32_to_fe4m3 = numpy.vectorize(float32_to_fe4m3)
        v_float32_to_fe5m2 = numpy.vectorize(float32_to_fe5m2)
        v_fe5m2_to_float32 = numpy.vectorize(fe5m2_to_float32)

        got = v_fe4m3_to_float32(v_float32_to_fe4m3(np_fp32))
        expected = numpy.array([0.46875, 0.46875, 0.5, 0.8125, 0.46875, 0.8125,
                                0.203125, 0.75, numpy.nan, 448.0, 448.0, -448.0],
                               dtype=numpy.float32)
        self.assertEqualArray(expected, got)
        got = v_fe5m2_to_float32(v_float32_to_fe5m2(np_fp32))
        expected = numpy.array([0.5, 0.5, 0.5, 0.875, 0.5, 0.875,
                                0.21875, 0.75, numpy.nan, numpy.inf,
                                numpy.inf, -numpy.inf],
                               dtype=numpy.float32)
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    TestF8().test_inf_nan()
    unittest.main(verbosity=2)
