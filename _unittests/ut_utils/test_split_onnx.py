"""
@brief      test log(time=11s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from onnx import ModelProto
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from onnxruntime import InferenceSession
from mlprodict.onnx_conv import to_onnx
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.utils.onnx_split import split_onnx, OnnxSplitting


class TestSplitOnnx(ExtTestCase):

    @ignore_warnings(ConvergenceWarning)
    def test_split_onnx(self):
        X = numpy.arange(30).reshape((-1, 3)).astype(numpy.float32) / 100
        y = numpy.arange(X.shape[0]).astype(numpy.float32)
        y = y.reshape((-1, 1))
        reg = MLPRegressor(hidden_layer_sizes=(5,), max_iter=2,
                           activation='logistic',
                           momentum=0, nesterovs_momentum=False,
                           alpha=0)
        reg.fit(X, y.ravel())

        onx = to_onnx(reg, X, target_opset=opset)
        split = OnnxSplitting(onx)
        segs = split.segments
        self.assertGreater(len(segs), 3)
        self.assertIn("OnnxSegment(", repr(segs[0]))
        total = 0
        for seg in split.segments:
            total += seg.size
        self.assertLess(len(onx.SerializeToString()), total)

        for n_parts in [2, 4]:
            with self.subTest(n_parts=n_parts):
                parts = split_onnx(onx, n_parts)
                self.assertEqual(len(parts), n_parts)
                for i, p in enumerate(parts):
                    if len(p.graph.input) == 0:
                        raise AssertionError(f"No input in part {i}\n{p}")
                    if len(p.graph.output) == 0:
                        raise AssertionError(f"No output in part {i}\n{p}")
                    if len(p.graph.node) == 0:
                        raise AssertionError(f"No node in part {i}\n{p}")
                    self.assertIsInstance(p, ModelProto)

                main = InferenceSession(onx.SerializeToString())
                rtp = [InferenceSession(p.SerializeToString()) for p in parts]

                expected = reg.predict(X)
                got = main.run(None, {'X': X})[0]
                self.assertEqualArray(expected, numpy.squeeze(got))

                feeds = {'X': X}
                for rt in rtp:
                    out = rt.run(None, feeds)[0]
                    n = rt.get_outputs()[0].name
                    feeds = {n: out}
                self.assertEqualArray(expected, numpy.squeeze(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
