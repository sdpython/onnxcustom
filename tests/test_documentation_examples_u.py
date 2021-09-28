"""
Tests examples from the documentation.
"""
import unittest
from distutils.version import StrictVersion
import os
import sys
import importlib
import subprocess
from datetime import datetime
import onnxruntime


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


class TestDocumentationExample_u(unittest.TestCase):

    def test_documentation_examples_u(self):

        this = os.path.abspath(os.path.dirname(__file__))
        onxc = os.path.normpath(os.path.join(this, '..'))
        pypath = os.environ.get('PYTHONPATH', None)
        sep = ";" if sys.platform == 'win32' else ':'
        pypath = "" if pypath in (None, "") else (pypath + sep)
        pypath += onxc
        os.environ['PYTHONPATH'] = pypath
        fold = os.path.normpath(os.path.join(this, '..', 'examples'))
        found = os.listdir(fold)
        tested = 0
        for name in sorted(found):
            if name >= "plot_u":
                break
            if 'lightgbm' in name:
                continue
            if 'training' in name:
                continue

            if '-v' in sys.argv or "--verbose" in sys.argv:
                if name.endswith('plot_bbegin_measure_time.py'):
                    if __name__ == "__main__":
                        print("%s: skip %r" % (
                            datetime.now().strftime("%d-%m-%y %H:%M:%S"),
                            name))
                    continue

            with self.subTest(name=name):
                if name.startswith("plot_") and name.endswith(".py"):
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
                        out, err = res
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
                            elif "No module named 'xgboost'" in st:
                                # xgboost not installed on CI
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
                                raise RuntimeError(
                                    "Example '{}' (cmd: {} - exec_prefix="
                                    "'{}') failed due to\n{}"
                                    "".format(name, cmds, sys.exec_prefix, st))
                    finally:
                        if sys.path[0] == fold:
                            del sys.path[0]
                    tested += 1
        if tested == 0:
            raise RuntimeError("No example was tested.")


if __name__ == "__main__":
    unittest.main()
