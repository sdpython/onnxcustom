"""
@brief      test log(time=60s)
"""
import unittest
import os
import sys
import importlib
import subprocess
from datetime import datetime
try:
    import onnxruntime.capi.ort_trainer as ortt
except ImportError:
    ortt = None
from pyquickhelper.pycode import skipif_circleci, ExtTestCase, skipif_appveyor
from pyquickhelper.texthelper import compare_module_version
from mlprodict import __version__ as mlp_version


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


class TestDocumentationExampleBenchmark(ExtTestCase):

    @unittest.skipIf(
        compare_module_version(mlp_version, "0.7.1642") <= 0,
        reason="plot_onnx was updated.")
    @unittest.skipIf(
        ortt is None, reason="onnxruntime-training not installed.")
    @skipif_circleci("stuck")
    @skipif_appveyor("too long")
    def test_documentation_examples(self):

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
            if 'benchmark' not in name:
                continue
            if 'orttraining' in name:
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
