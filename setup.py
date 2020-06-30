# -*- coding: utf-8 -*-
import os
from setuptools import setup
from setuptools import find_packages

######################
# beginning of setup
######################


here = os.path.dirname(__file__)
if here == "":
    here = '.'
packages = find_packages(where=here)
package_dir = {k: os.path.join(here, k.replace(".", "/")) for k in packages}
package_data = {
    "onnxcustom.data": ["*.csv"],
}

try:
    with open(os.path.join(here, "requirements.txt"), "r") as f:
        requirements = f.read().strip(' \n\r\t').split('\n')
except FileNotFoundError:
    requirements = []
if len(requirements) == 0 or requirements == ['']:
    requirements = []

try:
    with open(os.path.join(here, "readme.rst"), "r", encoding='utf-8') as f:
        long_description = "onnxcustom:" + f.read().split('onnxcustom:')[1]
except FileNotFoundError:
    long_description = ""

version_str = '0.0.1'
with open(os.path.join(here, 'onnxcustom/__init__.py'), "r") as f:
    line = [_ for _ in [_.strip("\r\n ")
                        for _ in f.readlines()]
            if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')

requires = ["cython", "fire", "numpy", "pandas", "scikit-learn",
            "scipy", "sympy", 'scikit-image']
with open(os.path.join(here, 'requirements.txt'), "r") as f:
    requires = [_.strip() for _ in f.readlines()]
    requires = [_ for _ in requires if _]

ext_modules = []


setup(name='onnxcustom',
      version=version_str,
      description="Custom ONNX operators and converters",
      long_description=long_description,
      author='Xavier Dupré',
      author_email='xavier.dupre@gmail.com',
      url='https://github.com/sdpython/onnxcustom',
      ext_modules=ext_modules,
      packages=packages,
      package_dir=package_dir,
      package_data=package_data,
      setup_requires=requires,
      install_requires=requires)
