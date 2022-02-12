"""
@brief      test log(time=9s)
"""
import unittest
import logging
from pyquickhelper.pycode import ExtTestCase
import numpy
from onnxruntime import InferenceSession
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=E0611
    Fail as OrtFail)
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxMul, OnnxIdentity)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt.validate.validate_latency import random_feed
from mlprodict.onnxrt import OnnxInference
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.training.grad_helper import onnx_derivative


class TestGradHelper(ExtTestCase):

    def check_runtime(self, onx, name):
        feeds = random_feed(onx.graph.input)
        n = 0
        for _, v in feeds.items():
            if v.shape[0] > 5:
                n += 1
        if n == 0:
            raise AssertionError(
                "No input with more than 5 rows: %r." % feeds)
        sess = InferenceSession(onx.SerializeToString())
        try:
            got = sess.run(None, feeds)
        except OrtFail as e:
            with open("fail_%s.onnx" % name, "wb") as f:
                f.write(onx.SerializeToString())
            raise AssertionError(
                "Unable to run onnx graph %r." % ("fail_%s.onnx" % name)) from e
        oinf = OnnxInference(onx)
        pygot = oinf.run(feeds)
        output_names = [o.name for o in onx.graph.output]
        self.assertGreater(len(output_names), 1)
        for i, o in enumerate(output_names):
            self.assertEqualArray(got[i], pygot[o])

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_grad_helper_keep_yield(self):
        opv = opset
        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, out_yield_op=False)
        types = set(n.op_type for n in new_onx.graph.node)  # pylint: disable=E1101
        self.assertIn('YieldOp', types)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_grad_helper(self):
        opv = opset
        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx)
        self.check_runtime(new_onx, 'test_grad_helper')

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_grad_helper_mul(self):
        opv = opset
        xi = OnnxIdentity('X', op_version=opv)
        node = OnnxMul(xi, xi, op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx)
        self.check_runtime(new_onx, 'test_grad_helper_mul')

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_grad_helper_noweight(self):
        opv = opset
        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, weights=[])
        self.check_runtime(new_onx, 'test_grad_helper_noweight')


if __name__ == "__main__":
    unittest.main()
