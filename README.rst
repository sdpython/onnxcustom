
.. image:: https://circleci.com/gh/sdpython/onnxcustom/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/onnxcustom/tree/master

.. image:: https://travis-ci.org/sdpython/onnxcustom.svg?branch=master
    :target: https://travis-ci.org/sdpython/onnxcustom
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/ffne8nhh96jdqo4p?svg=true
    :target: https://ci.appveyor.com/project/sdpython/onnxcustom
    :alt: Build Status Windows

.. image:: https://codecov.io/gh/sdpython/onnxcustom/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/sdpython/onnxcustom

.. image:: https://badge.fury.io/py/onnxcustom.svg
    :target: http://badge.fury.io/py/onnxcustom

.. image:: http://img.shields.io/github/issues/sdpython/onnxcustom.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/onnxcustom/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://img.shields.io/github/repo-size/sdpython/onnxcustom
    :target: https://github.com/sdpython/onnxcustom/
    :alt: size

onnxcustom: custom ONNX
=======================

.. image:: https://raw.githubusercontent.com/sdpython/onnxcustom/master/doc/_static/logo.png
    :width: 50

`documentation <http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/index.html>`_

Tools, tries about COVID epidemics.
The module must be compiled to be used inplace:

::

    python setup.py build_ext --inplace

Generate the setup in subfolder ``dist``:

::

    python setup.py sdist

Generate the documentation in folder ``dist/html``:

::

    python -m sphinx -T -b html doc dist/html

Run the unit tests:

::

    python -m unittest discover tests

Or:

::

    python -m pytest
    
To check style:

::

    python -m flake8 onnxcustom tests examples

The function *check* or the command line ``python -m onnxcustom check``
checks the module is properly installed and returns processing
time for a couple of functions or simply:

::

    import onnxcustom
    onnxcustom.check()
