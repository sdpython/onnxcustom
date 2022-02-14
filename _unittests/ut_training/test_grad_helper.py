# pylint: disable=E1101
"""
@brief      test log(time=9s)
"""
import unittest
import logging
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
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
from onnxcustom.training.grad_helper import onnx_derivative, DerivativeOptions


class TestGradHelper(ExtTestCase):

    def check_runtime(self, onx, name, decimal=5, verbose=False):
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
            self.assertEqualArray(got[i], pygot[o], decimal=decimal)
        if verbose:
            print("%s - input=%r output=%r" % (
                name, [o.name for o in onx.graph.input],
                [o.name for o in onx.graph.output]))
            with open("verbose_%s.onnx" % name, "wb") as f:
                f.write(onx.SerializeToString())

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)
        ExtTestCase.setUpClass()

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings(DeprecationWarning)
    def test_grad_helper_keep_yield(self):
        opv = opset
        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, options=DerivativeOptions.KeepYieldOp)
        types = set(
            n.op_type for n in new_onx.graph.node)
        self.assertIn('YieldOp', types)
        with open("verbose_%s.onnx" % 'yield', "wb") as f:
            f.write(new_onx.SerializeToString())

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings(DeprecationWarning)
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
    @ignore_warnings(DeprecationWarning)
    def test_grad_helper_nooutput(self):
        opv = opset
        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, options=DerivativeOptions.KeepOutputs)
        self.check_runtime(new_onx, 'test_grad_helper_nooutput')

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings(DeprecationWarning)
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
    @ignore_warnings(DeprecationWarning)
    def test_grad_helper_noweight(self):
        opv = opset
        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, weights=[])
        self.check_runtime(new_onx, 'test_grad_helper_noweight')

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings(DeprecationWarning)
    def test_grad_helper_fillgrad(self):
        opv = opset
        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        self.assertRaise(
            lambda: onnx_derivative(
                onx, weights=[], options=DerivativeOptions.FillGrad),
            ValueError)
        new_onx = onnx_derivative(
            onx, weights=[], options=(
                DerivativeOptions.FillGrad | DerivativeOptions.KeepOutputs))
        input_names = set(i.name for i in new_onx.graph.input)
        self.assertNotIn('Y_grad', input_names)
        self.check_runtime(new_onx, 'test_grad_helper_fillgrad', verbose=False)

    @ignore_warnings(DeprecationWarning)
    def test_grad_helper_exc(self):
        opv = opset
        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        self.assertRaise(
            lambda: onnx_derivative(onx, weights=[], options=1),
            TypeError)


if __name__ == "__main__":
    unittest.main()
