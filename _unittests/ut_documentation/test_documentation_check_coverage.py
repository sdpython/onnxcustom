# -*- coding: utf-8 -*-
"""
@brief      test log(time=2800s)
"""
import os
import unittest
from pyquickhelper.pycode import ExtTestCase, skipif_appveyor


class TestDocumentationCheckCoverage(ExtTestCase):

    @skipif_appveyor("not relevant")
    def test_examples_coverage(self):
        with open(
                os.path.join(os.path.dirname(__file__),
                             "_test_example.txt"), "r", encoding='utf-8') as f:
            lines = f.read().split('\n')

        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(
            os.path.join(this, '..', '..', '_doc', 'examples'))
        found = os.listdir(fold)

        done = set(_ for _ in lines if os.path.splitext(_)[-1] == '.py')
        found = set(_ for _ in found
                    if (os.path.splitext(_)[-1] == '.py' and
                        _.startswith('plot_')))
        if len(done) != len(found):
            missing = found - done
            raise AssertionError(
                "Following examples were not tested:\n%s."
                "" % "\n".join(sorted(missing)))


if __name__ == "__main__":
    unittest.main()
