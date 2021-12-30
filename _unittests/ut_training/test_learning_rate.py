"""
@brief      test log(time=3s)
"""

import unittest
from pyquickhelper.pycode import ExtTestCase
from onnxcustom.training.sgd_learning_rate import LearningRateSGD


class TestLearningRate(ExtTestCase):

    def is_decreased(self, series):
        for i in range(1, len(series)):
            if series[i] >= series[i - 1]:
                raise AssertionError(
                    "Not decreasing at index %d - %r." % (
                        i, series[i - 1: i + 1]))

    def test_learning_rate_sgd_regressor_default(self):
        cllr = LearningRateSGD()
        val = list(cllr.loop())
        self.assertEqual(len(val), 1000)
        self.is_decreased(val)
        self.assertEqual(val[0], 0.01)
        self.assertGreater(val[-1], 0.001)

    def test_learning_rate_sgd_regressor_exc(self):
        self.assertRaise(
            lambda: LearningRateSGD(learning_rate='EXC'),
            ValueError)

    def test_learning_rate_sgd_regressor_optimal(self):
        cllr = LearningRateSGD(learning_rate='optimal')
        val = list(cllr.loop())
        self.assertEqual(len(val), 1000)
        self.is_decreased(val)
        self.assertEqual(val[0], 0.01)
        self.assertGreater(val[-1], 0.009)

    def test_learning_rate_sgd_regressor_constant(self):
        cllr = LearningRateSGD(learning_rate='constant')
        val = list(cllr.loop())
        self.assertEqual(len(val), 1000)
        self.assertEqual(val[0], 0.01)
        self.assertEqual(val[-1], val[0])

    def test_learning_rate_sgd_exc(self):
        self.assertRaise(
            lambda: LearningRateSGD(learning_rate='CST'),
            ValueError)


if __name__ == "__main__":
    unittest.main()
