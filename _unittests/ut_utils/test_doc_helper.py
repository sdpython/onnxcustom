"""
@brief      test log(time=5s)
"""
import os
import unittest
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from onnxcustom.utils.doc_helper import fix_link_operator_md


class TestDocHelper(ExtTestCase):

    def test_doc_helper_op(self):
        temp = get_temp_folder(__file__, 'temp_doc_helper_op')
        data = os.path.join(temp, "..", "..", "..", "_doc", "sphinxdoc",
                            "source", "onnxmd", "onnx_docs",
                            "Operators.md")
        new_content = fix_link_operator_md(data)
        output = os.path.join(temp, "Operators.md")
        with open(output, "w", encoding="utf-8") as f:
            f.write(new_content)
        self.assertExists(output)
        self.assertIn(
            '|[Mul](#a-name-mul-a-a-name-mul-mul-a)|', new_content)
        self.assertNotIn(
            '|<a href="#a-name-mul-a-a-name-mul-mul-a">Mul</a>|',
            new_content)

    def test_doc_helper_op_ml(self):
        temp = get_temp_folder(__file__, 'temp_doc_helper_op_ml')
        data = os.path.join(temp, "..", "..", "..", "_doc", "sphinxdoc",
                            "source", "onnxmd", "onnx_docs",
                            "Operators-ml.md")
        new_content = fix_link_operator_md(data)
        output = os.path.join(temp, "Operators-ml.md")
        with open(output, "w", encoding="utf-8") as f:
            f.write(new_content)
        self.assertExists(output)
        self.assertNotIn(
            '|<a href="#ai.onnx.ml.SVMRegressor">ai.onnx.ml.SVMRegressor</a>|',
            new_content)
        self.assertIn(
            '|[ai-onnx-ml-SVMRegressor](#a-name-ai-onnx-ml-svmregressor-a-a-name'
            '-ai-onnx-ml-svmregressor-ai-onnx-ml-svmregressor-a)|', new_content)


if __name__ == "__main__":
    unittest.main()
