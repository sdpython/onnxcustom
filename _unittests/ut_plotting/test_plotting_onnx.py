# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import os
import warnings
import unittest
import numpy
from pyquickhelper.pycode import (
    ExtTestCase, skipif_travis, skipif_circleci, get_temp_folder)
from skl2onnx.algebra.onnx_ops import OnnxConcat  # pylint: disable=E0611
from skl2onnx.common.data_types import FloatTensorType
from onnxcustom.plotting.plotting_onnx import plot_onnxs


class TestPlotOnnx(ExtTestCase):

    @skipif_travis('graphviz is not installed')
    @skipif_circleci('graphviz is not installed')
    def test_plot_onnx(self):

        cst = numpy.array([[1, 2]], dtype=numpy.float32)
        onx = OnnxConcat('X', 'Y', cst, output_names=['Z'],
                         op_version=12)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        Y = numpy.array([[8, 9], [10, 11], [12, 13]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=12)

        import matplotlib.pyplot as plt
        self.assertRaise(lambda: plot_onnxs(*[]), ValueError)

        try:
            ax = plot_onnxs(model_def, title="GRAPH")
        except FileNotFoundError as e:
            if "No such file or directory: 'dot'" in str(e):
                warnings.warn(
                    "Unable to test the dot syntax, dot is mssing", UserWarning)
                return
            raise e
        self.assertNotEmpty(ax)
        plt.close('all')

    @skipif_travis('graphviz is not installed')
    @skipif_circleci('graphviz is not installed')
    def test_plot_onnx2(self):

        cst = numpy.array([[1, 2]], dtype=numpy.float32)
        onx = OnnxConcat('X', 'Y', cst, output_names=['Z'],
                         op_version=12)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        Y = numpy.array([[8, 9], [10, 11], [12, 13]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=12)

        import matplotlib.pyplot as plt
        ax = numpy.array([0])
        self.assertRaise(
            lambda: plot_onnxs(model_def, model_def, ax=ax), ValueError)

        try:
            ax = plot_onnxs(model_def, model_def, title=["GRAPH1", "GRAPH2"])
        except FileNotFoundError as e:
            if "No such file or directory: 'dot'" in str(e):
                warnings.warn(
                    "Unable to test the dot syntax, dot is mssing", UserWarning)
                return
            raise e
        self.assertNotEmpty(ax)
        try:
            ax = plot_onnxs(model_def, model_def, title="GRAPH1")
        except FileNotFoundError as e:
            if "No such file or directory: 'dot'" in str(e):
                warnings.warn(
                    "Unable to test the dot syntax, dot is mssing", UserWarning)
                return
            raise e
        self.assertNotEmpty(ax)
        if __name__ == "__main__":
            temp = get_temp_folder(__file__, "temp_plot_onnx2")
            img = os.path.join(temp, "img.png")
            plt.savefig(img)
            plt.show()
        plt.close('all')


if __name__ == "__main__":
    unittest.main()
