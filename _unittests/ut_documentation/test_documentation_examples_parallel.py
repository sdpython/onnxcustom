"""
@brief      test log(time=60s)
"""
import unittest
import os
import sys
import importlib
import subprocess
from datetime import datetime
from pyquickhelper.pycode import ExtTestCase, skipif_appveyor
from pyquickhelper.filehelper import synchronize_folder


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


class TestDocumentationParallel(ExtTestCase):

    @skipif_appveyor("too long")
    def test_documentation_examples_parallel(self):

        this = os.path.abspath(os.path.dirname(__file__))
        onxc = os.path.normpath(os.path.join(this, '..', '..'))
        pypath = os.environ.get('PYTHONPATH', None)
        sep = ";" if sys.platform == 'win32' else ':'
        pypath = "" if pypath in (None, "") else (pypath + sep)
        pypath += onxc
        os.environ['PYTHONPATH'] = pypath
        fold = os.path.normpath(
            os.path.join(this, '..', '..', '_doc', 'examples'))
        synchronize_folder(os.path.join(fold, "data"), "data",
                           no_deletion=True, create_dest=True)
        found = os.listdir(fold)
        tested = 0
        for name in sorted(found):
            if 'parallel' not in name:
                continue
            if not name.startswith("plot_") or not name.endswith(".py"):
                continue

            if '-v' in sys.argv or "--verbose" in sys.argv:
                if name.endswith('plot_bbegin_measure_time.py'):
                    if __name__ == "__main__":
                        print("%s: skip %r" % (
                            datetime.now().strftime("%d-%m-%y %H:%M:%S"),
                            name))
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
