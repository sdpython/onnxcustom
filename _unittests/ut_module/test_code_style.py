"""
@brief      test log(time=0s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import check_pep8, ExtTestCase, skipif_circleci


class TestCodeStyle(ExtTestCase):
    """Test style."""

    def test_style_src(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        src_ = os.path.normpath(os.path.join(thi, "..", "..", "onnxcustom"))
        check_pep8(src_, fLOG=fLOG,
                   pylint_ignore=(
                        'C0103', 'C1801', 'R0201', 'R1705', 'W0108', 'W0613',
                        'W0201', 'W0221', 'E0632', 'R1702', 'W0212', 'W0223',
                        'W0107', "R1720", 'R1732', 'C0209'),
                   skip=[])

    def test_style_test(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        test = os.path.normpath(os.path.join(thi, "..", ))
        check_pep8(test, fLOG=fLOG, neg_pattern="temp_.*",
                   pylint_ignore=(
                        'C0103', 'C1801', 'R0201', 'R1705', 'W0108', 'W0613',
                        'C0111', 'W0107', 'C0111', 'R1702', 'C0415', "R1720",
                        'R1732', 'C0209'),
                   skip=[])


if __name__ == "__main__":
    unittest.main()
