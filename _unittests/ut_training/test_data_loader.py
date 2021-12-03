"""
@brief      test log(time=3s)
"""

import unittest
import io
import pickle
import numpy
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from pyquickhelper.pycode import ExtTestCase, skipif_circleci
from sklearn.datasets import make_regression
from onnxcustom.training.data_loader import OrtDataLoader


class TestDataLoader(ExtTestCase):

    @skipif_circleci("bizarre")
    def test_ort_data_loader(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        data = OrtDataLoader(X, y, batch_size=5)
        n = 0
        for it in data.iter_ortvalue():
            x, y = it
            self.assertIsInstance(x, C_OrtValue)
            self.assertIsInstance(y, C_OrtValue)
            self.assertEqual(x.shape()[0], 5)
            self.assertEqual(x.shape()[1], 10)
            self.assertEqual(y.shape()[0], 5)
            n += 1
        self.assertEqual(n, 20)
        self.assertStartsWith("OrtDataLoader(...", repr(data))
        self.assertIsInstance(data.data_np, tuple)
        self.assertIsInstance(data.data_ort, tuple)
        self.assertEqual(
            data.desc,
            [((100, 10), numpy.float64), ((100, 1), numpy.float64)])

    @skipif_circleci("bizarre")
    def test_ort_data_loader_numpy(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        data = OrtDataLoader(X, y, batch_size=5)
        n = 0
        for it in data.iter_numpy():
            x, y = it
            self.assertIsInstance(x, numpy.ndarray)
            self.assertIsInstance(y, numpy.ndarray)
            self.assertEqual(x.shape[0], 5)
            self.assertEqual(x.shape[1], 10)
            self.assertEqual(y.shape[0], 5)
            n += 1
        self.assertEqual(n, 20)
        self.assertStartsWith("OrtDataLoader(...", repr(data))
        self.assertIsInstance(data.data_np, tuple)
        self.assertIsInstance(data.data_ort, tuple)
        self.assertEqual(
            data.desc,
            [((100, 10), numpy.float64), ((100, 1), numpy.float64)])

    @skipif_circleci("bizarre")
    def test_ort_data_loader_pickle(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        data0 = OrtDataLoader(X, y, batch_size=5)
        st = io.BytesIO()
        pickle.dump(data0, st)
        st2 = io.BytesIO(st.getvalue())
        data = pickle.load(st2)
        n = 0
        for it in data.iter_ortvalue():
            x, y = it
            self.assertIsInstance(x, C_OrtValue)
            self.assertIsInstance(y, C_OrtValue)
            self.assertEqual(x.shape()[0], 5)
            self.assertEqual(x.shape()[1], 10)
            self.assertEqual(y.shape()[0], 5)
            n += 1
        self.assertEqual(n, 20)
        self.assertStartsWith("OrtDataLoader(...", repr(data))
        self.assertIsInstance(data.data_np, tuple)
        self.assertIsInstance(data.data_ort, tuple)
        self.assertEqual(
            data.desc,
            [((100, 10), numpy.float64), ((100, 1), numpy.float64)])

    @skipif_circleci("bizarre")
    def test_ort_data_loader_compare(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        data = OrtDataLoader(X, y, batch_size=5, random_iter=False)
        values_ort = []
        values_np = []
        for it in data.iter_ortvalue():
            values_ort.append(it)
        for it in data.iter_numpy():
            values_np.append(it)
        self.assertEqual(len(values_ort), len(values_np))
        for it, (o, n) in enumerate(zip(values_ort, values_np)):
            self.assertEqual(len(o), len(n))
            self.assertEqual(len(o), 2)
            ov = (o[0].numpy(), o[1].numpy())
            self.assertEqualArray(ov[0], n[0])
            self.assertEqualArray(ov[1], n[1])
            i = it * 5
            self.assertEqualArray(X[i: i + 5], n[0])
            self.assertEqualArray(y[i: i + 5], n[1].ravel())


if __name__ == "__main__":
    unittest.main()
