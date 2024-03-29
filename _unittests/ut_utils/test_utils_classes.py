"""
@brief      test log(time=3s)
"""

import unittest
from onnxcustom.utils.imagenet_classes import (
    class_names, get_class_names)


class TestUtilsClasses(unittest.TestCase):

    def test_classes(self):
        cl = class_names
        self.assertIsInstance(cl, dict)
        self.assertEqual(len(cl), 1000)

    def test_get_classes(self):
        cl = get_class_names()
        self.assertIsInstance(cl, dict)
        self.assertEqual(len(cl), 1000)


if __name__ == "__main__":
    unittest.main()
