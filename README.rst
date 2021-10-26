
.. image:: https://circleci.com/gh/sdpython/onnxcustom/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/onnxcustom/tree/master

.. image:: https://travis-ci.com/sdpython/onnxcustom.svg?branch=master
    :target: https://app.travis-ci.com/github/sdpython/onnxcustom
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/a3sn45a2fayoxb5q?svg=true
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

.. image:: https://pepy.tech/badge/onnxcustom/month
    :target: https://pepy.tech/project/onnxcustom/month
    :alt: Downloads

.. image:: https://img.shields.io/github/forks/sdpython/onnxcustom.svg
    :target: https://github.com/sdpython/onnxcustom/
    :alt: Forks

.. image:: https://img.shields.io/github/stars/sdpython/onnxcustom.svg
    :target: https://github.com/sdpython/onnxcustom/
    :alt: Stars

.. image:: https://img.shields.io/github/repo-size/sdpython/onnxcustom
    :target: https://github.com/sdpython/onnxcustom/
    :alt: size

onnxcustom: custom ONNX
=======================

.. image:: https://raw.githubusercontent.com/sdpython/deeponnxcustom/master/_doc/sphinxdoc/source/phdoc_static/project_ico.png
    :width: 50

`documentation <http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/index.html>`_

Tutorial on how to convert machine learned models into ONNX,
implement your own converter or runtime.
The module must be compiled to be used inplace:

::

    python setup.py build_ext --inplace

Generate the setup in subfolder ``dist``:

::

    python setup.py sdist

Generate the documentation in folder ``dist/html``:

::

    python setup.py build_sphinx

Run the unit tests:

::

    python setup.py unittests

To check style:

::

    python -m flake8 onnxcustom tests examples

The function *check* or the command line ``python -m onnxcustom check``
checks the module is properly installed and returns processing
time for a couple of functions or simply:

::

    import onnxcustom
    onnxcustom.check()

This tutorial has been merged into `sklearn-onnx documentation
<http://onnx.ai/sklearn-onnx/index_tutorial.html>`_.
