# -*- coding: utf-8 -*-
"""
@brief      test log(time=80s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, skipif_circleci, skipif_appveyor
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
import onnxcustom


class TestDocumentationNotebooksPython(ExtTestCase):

    def setUp(self):
        import jyquickhelper  # pylint: disable=C0415
        self.assertTrue(jyquickhelper is not None)

    @skipif_circleci("stuck")
    @skipif_appveyor("too long")
    def test_notebook_artificiel(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertTrue(onnxcustom is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        test_notebook_execution_coverage(
            __file__, "tree", folder, 'onnxcustom', copy_files=[], fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
