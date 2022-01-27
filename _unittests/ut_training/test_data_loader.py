"""
@brief      test log(time=13s)
"""

import unittest
import io
import pickle
import numpy
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import make_regression
from onnxcustom.training.data_loader import OrtDataLoader


class TestDataLoader(ExtTestCase):

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

    def test_ort_data_loader_101(self):
        X, y = make_regression(  # pylint: disable=W0632
            101, n_features=10, bias=2)
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
        self.assertEqual(n, 21)
        self.assertStartsWith("OrtDataLoader(...", repr(data))
        self.assertIsInstance(data.data_np, tuple)
        self.assertIsInstance(data.data_ort, tuple)
        self.assertEqual(
            data.desc,
            [((101, 10), numpy.float64), ((101, 1), numpy.float64)])

    def test_ort_data_loader_11(self):
        X, y = make_regression(  # pylint: disable=W0632
            11, n_features=10, bias=2)
        data = OrtDataLoader(X, y, batch_size=15)
        n = 0
        for it in data.iter_ortvalue():
            x, y = it
            self.assertIsInstance(x, C_OrtValue)
            self.assertIsInstance(y, C_OrtValue)
            self.assertEqual(x.shape()[0], 11)
            self.assertEqual(x.shape()[1], 10)
            self.assertEqual(y.shape()[0], 11)
            n += 1
        self.assertEqual(n, 1)
        self.assertStartsWith("OrtDataLoader(...", repr(data))
        self.assertIsInstance(data.data_np, tuple)
        self.assertIsInstance(data.data_ort, tuple)
        self.assertEqual(
            data.desc,
            [((11, 10), numpy.float64), ((11, 1), numpy.float64)])

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

    def test_ort_data_loader_numpy_exc(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        self.assertRaise(
            lambda: OrtDataLoader(X, y, batch_size=5, device='cpu2'),
            Exception)

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

    def test_ort_data_loader_w(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        data = OrtDataLoader(X, y, w, batch_size=5)
        n = 0
        for it in data.iter_ortvalue():
            x, y, w = it
            self.assertIsInstance(x, C_OrtValue)
            self.assertIsInstance(y, C_OrtValue)
            self.assertIsInstance(w, C_OrtValue)
            self.assertEqual(x.shape()[0], 5)
            self.assertEqual(x.shape()[1], 10)
            self.assertEqual(y.shape()[0], 5)
            self.assertEqual(w.shape()[0], 5)
            n += 1
        self.assertEqual(n, 20)
        self.assertStartsWith("OrtDataLoader(...", repr(data))
        self.assertIsInstance(data.data_np, tuple)
        self.assertIsInstance(data.data_ort, tuple)
        self.assertEqual(
            data.desc,
            [((100, 10), numpy.float64), ((100, 1), numpy.float64),
             ((100, ), numpy.float64)])

    def test_ort_data_loader_numpy_w(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        data = OrtDataLoader(X, y, w, batch_size=5)
        n = 0
        for it in data.iter_numpy():
            x, y, w = it
            self.assertIsInstance(x, numpy.ndarray)
            self.assertIsInstance(y, numpy.ndarray)
            self.assertIsInstance(w, numpy.ndarray)
            self.assertEqual(x.shape[0], 5)
            self.assertEqual(x.shape[1], 10)
            self.assertEqual(y.shape[0], 5)
            self.assertEqual(w.shape[0], 5)
            n += 1
        self.assertEqual(n, 20)
        self.assertStartsWith("OrtDataLoader(...", repr(data))
        self.assertIsInstance(data.data_np, tuple)
        self.assertIsInstance(data.data_ort, tuple)
        self.assertEqual(
            data.desc,
            [((100, 10), numpy.float64), ((100, 1), numpy.float64),
             ((100, ), numpy.float64)])

    def test_ort_data_loader_numpy_exc_w(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        self.assertRaise(
            lambda: OrtDataLoader(X, y, w, batch_size=5, device='cpu2'),
            Exception)

    def test_ort_data_loader_pickle_w(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        data0 = OrtDataLoader(X, y, w, batch_size=5)
        st = io.BytesIO()
        pickle.dump(data0, st)
        st2 = io.BytesIO(st.getvalue())
        data = pickle.load(st2)
        n = 0
        for it in data.iter_ortvalue():
            x, y, w = it
            self.assertIsInstance(x, C_OrtValue)
            self.assertIsInstance(y, C_OrtValue)
            self.assertIsInstance(w, C_OrtValue)
            self.assertEqual(x.shape()[0], 5)
            self.assertEqual(x.shape()[1], 10)
            self.assertEqual(y.shape()[0], 5)
            self.assertEqual(w.shape()[0], 5)
            n += 1
        self.assertEqual(n, 20)
        self.assertStartsWith("OrtDataLoader(...", repr(data))
        self.assertIsInstance(data.data_np, tuple)
        self.assertIsInstance(data.data_ort, tuple)
        self.assertEqual(
            data.desc,
            [((100, 10), numpy.float64), ((100, 1), numpy.float64),
             ((100, ), numpy.float64)])

    def test_ort_data_loader_compare_w(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        data = OrtDataLoader(X, y, w, batch_size=5, random_iter=False)
        values_ort = []
        values_np = []
        for it in data.iter_ortvalue():
            values_ort.append(it)
        for it in data.iter_numpy():
            values_np.append(it)
        self.assertEqual(len(values_ort), len(values_np))
        for it, (o, n) in enumerate(zip(values_ort, values_np)):
            self.assertEqual(len(o), len(n))
            self.assertEqual(len(o), 3)
            ov = (o[0].numpy(), o[1].numpy(), o[2].numpy())
            self.assertEqualArray(ov[0], n[0])
            self.assertEqualArray(ov[1], n[1])
            self.assertEqualArray(ov[2], n[2])
            i = it * 5
            self.assertEqualArray(X[i: i + 5], n[0])
            self.assertEqualArray(y[i: i + 5], n[1].ravel())
            self.assertEqualArray(w[i: i + 5], n[2].ravel())


if __name__ == "__main__":
    unittest.main()
