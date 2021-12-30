"""
@brief      test log(time=60s)
"""
import unittest
from distutils.version import StrictVersion
import os
import sys
import importlib
import subprocess
from datetime import datetime
import onnxruntime
from pyquickhelper.pycode import ExtTestCase


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(
        module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}', cwd='{}'.".format(
                module_name, module_file_path,
                os.path.abspath(__file__)))
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExampleLightgbm(ExtTestCase):

    def test_documentation_examples_lightgbm(self):

        this = os.path.abspath(os.path.dirname(__file__))
        onxc = os.path.normpath(os.path.join(this, '..', '..'))
        pypath = os.environ.get('PYTHONPATH', None)
        sep = ";" if sys.platform == 'win32' else ':'
        pypath = "" if pypath in (None, "") else (pypath + sep)
        pypath += onxc
        os.environ['PYTHONPATH'] = pypath
        fold = os.path.normpath(
            os.path.join(this, '..', '..', '_doc', 'examples'))
        found = os.listdir(fold)
        tested = 0
        for name in sorted(found):
            if 'lightgbm' not in name:
                continue
            if 'reg' in name:
                from pyquickhelper.pycode.ci_helper import (
                    is_travis_or_appveyor)
                if is_travis_or_appveyor() == 'circleci':
                    # stuck
                    continue

            if '-v' in sys.argv or "--verbose" in sys.argv:
                if name.endswith('plot_bbegin_measure_time.py'):
                    if __name__ == "__main__":
                        print("%s: skip %r" % (
                            datetime.now().strftime("%d-%m-%y %H:%M:%S"),
                            name))
                    continue
            if (name == "plot_pipeline_lightgbm.py" and
                    StrictVersion(onnxruntime.__version__) <
                        StrictVersion('1.0.0')):
                continue
            if not name.startswith("plot_") or not name.endswith(".py"):
                continue

            with self.subTest(name=name):
                if __name__ == "__main__" or "-v" in sys.argv:
                    print("%s: run %r" % (
                        datetime.now().strftime("%d-%m-%y %H:%M:%S"),
                        name))
                sys.path.insert(0, fold)
                try:
                    mod = import_source(fold, os.path.splitext(name)[0])
                    assert mod is not None
                except FileNotFoundError:
                    # try another way
                    cmds = [sys.executable, "-u",
                            os.path.join(fold, name)]
                    p = subprocess.Popen(
                        cmds, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
                    res = p.communicate()
                    _, err = res
                    st = err.decode('ascii', errors='ignore')
                    if len(st) > 0 and 'Traceback' in st:
                        if "No such file or directory: 'dot': 'dot'" in st:
                            # dot not installed, this part
                            # is tested in onnx framework
                            pass
                        elif '"dot" not found in path.' in st:
                            # dot not installed, this part
                            # is tested in onnx framework
                            pass
                        elif ("cannot import name 'LightGbmModelContainer'"
                                " from 'onnxmltools.convert.common."
                                "_container'") in st:
                            # onnxmltools not recent enough
                            pass
                        elif ('Please fix either the inputs or '
                                'the model.') in st:
                            # onnxruntime datasets changed in master
                            # branch, still the same in released
                            # version on pypi
                            pass
                        elif 'dot: graph is too large' in st:
                            # graph is too big
                            pass
                        else:
                            raise RuntimeError(  # pylint: disable=W0707
                                "Example '{}' (cmd: {} - exec_prefix="
                                "'{}') failed due to\n{}"
                                "".format(name, cmds, sys.exec_prefix, st))
                finally:
                    if sys.path[0] == fold:
                        del sys.path[0]
                with open(
                        os.path.join(os.path.dirname(__file__),
                                     "_test_example.txt"), "a",
                        encoding='utf-8') as f:
                    f.write(name + "\n")
                tested += 1
        if tested == 0:
            raise RuntimeError("No example was tested.")


if __name__ == "__main__":
    unittest.main()
