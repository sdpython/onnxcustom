"""
@brief      test log(time=60s)
"""
import unittest
import os
from pyquickhelper.pycode import ExtTestCase


class TestDocumentationExampleShow(ExtTestCase):

    def test_documentation_examples_show(self):

        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(
            os.path.join(this, '..', '..', '_doc', 'examples'))
        found = os.listdir(fold)
        tested = 0
        for name in sorted(found):
            if not name.startswith("plot_") or not name.endswith(".py"):
                continue

            with self.subTest(name=name):
                full_name = os.path.join(fold, name)
                with open(full_name, "r", encoding="utf-8") as f:
                    content = f.read()
                if "plt.show()" in content and "# plt.show()" not in content:
                    raise AssertionError(
                        "plt.show() not found in %r." % name)
                tested += 1
        if tested == 0:
            raise RuntimeError("No example was tested.")


if __name__ == "__main__":
    unittest.main()
