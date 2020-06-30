
Developers' corner
==================

.. contents::
    :local:

Build, documentation, unittests
+++++++++++++++++++++++++++++++

Build the module inplace:

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

Setup
+++++

Instructions can be found in CI files:

* `Windows <https://github.com/sdpython/onnxcustom/blob/master/appveyor.yml>`_
* `Linux (Debian) <https://github.com/sdpython/onnxcustom/blob/master/.circleci/config.yml>`_
* `Linux (Ubuntu) <https://github.com/sdpython/onnxcustom/blob/master/.travis.yml>`_
* `Mac OSX <https://github.com/sdpython/onnxcustom/blob/master/azure-pipelines.yml#L50>`_

Full script to build
++++++++++++++++++++

::

    echo --CLONE--
    git clone -b master --single-branch https://github.com/sdpython/onnxcustom.git --recursive
    cd onnxcustom

    echo --PIP--
    python -m pip install -r requirements.txt || exit 1
    python -m pip install -r requirements-dev.txt || exit 1

    echo --WHEEL--
    python -u setup.py build_ext --inplace || exit 1

    echo --TEST--
    python -m pytest -v -v || exit 1
    python -m coverage run  --omit=tests/test_*.py -m unittest discover tests -v -v || exit 1
    python -m coverage html -d dist/html/coverage.html --include **/onnxcustom/** || exit 1
    python -m flake8 . || exit 1

    echo --WHEEL--
    python -u setup.py bdist_wheel || exit 1

    echo --SPEED--
    python -m onnxcustom check || exit 1

    echo --DOC--
    python setup.py install || exit 1
    python -m sphinx -b html doc dist/html || exit 1

    echo --END--
    cd ..
