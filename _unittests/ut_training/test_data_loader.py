"""
@brief      test log(time=3s)
"""

import unittest
import io
import pickle
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import make_regression
from onnxruntime import OrtValue
from onnxcustom.training.data_loader import OrtDataLoader


class TestDataLoadeer(ExtTestCase):

    def test_ort_data_loader(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        data = OrtDataLoader(X, y, batch_size=5)
        n = 0
        for it in data:
            x, y = it
            self.assertIsInstance(x, OrtValue)
            self.assertIsInstance(y, OrtValue)
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

    def test_ort_data_loader_pickle(self):
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        data0 = OrtDataLoader(X, y, batch_size=5)
        st = io.BytesIO()
        pickle.dump(data0, st)
        st2 = io.BytesIO(st.getvalue())
        data = pickle.load(st2)
        n = 0
        for it in data:
            x, y = it
            self.assertIsInstance(x, OrtValue)
            self.assertIsInstance(y, OrtValue)
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


if __name__ == "__main__":
    unittest.main()
