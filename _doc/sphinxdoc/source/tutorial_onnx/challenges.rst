
Challenges
==========

.. contents::
    :local:

Opsets
++++++

Tricks learned from experience
++++++++++++++++++++++++++++++

Discrepancies
~~~~~~~~~~~~~

Types, order of computation, parallelization

IsolationForest trick
~~~~~~~~~~~~~~~~~~~~~

Contribute
++++++++++

`onnx repository <https://github.com/onnx/onnx>`_ must be forked and cloned.

Build
~~~~~

The windows build requires conda. The following steps might not be up to date.
Folder `onnx/.azure-pipelines
<https://github.com/onnx/onnx/tree/master/.azure-pipelines>`_
contains the latest instructions.

**Windows**

The build is easier with :epkg:`Anaconda`. First: create an environment.
It must be done only once.

::

    conda create --yes --quiet --name py3.9 python=3.9
    conda install -n py3.9 -y -c conda-forge numpy libprotobuf=3.16.0

Then build the package:

::

    git submodule update --init --recursive
    set ONNX_BUILD_TESTS=1
    set ONNX_ML=$(onnx_ml)
    set CMAKE_ARGS=-DONNX_USE_PROTOBUF_SHARED_LIBS=ON -DONNX_USE_LITE_PROTO=ON -DONNX_WERROR=ON

    python setup.py -q install
    python setup.py bdist_wheel

The package can now be installed.

Build the documentation
~~~~~~~~~~~~~~~~~~~~~~~

The package must be built first (see previous section).

::

    set ONNX_BUILD_TESTS=1
    set ONNX_ML=$(onnx_ml)
    set CMAKE_ARGS=-DONNX_USE_PROTOBUF_SHARED_LIBS=ON -DONNX_USE_LITE_PROTO=ON -DONNX_WERROR=ON

    python onnx\gen_proto.py -l
    python onnx\gen_proto.py -l --ml
    python setup.py develop
    python onnx\backend\test\cmd_tools.py generate-data
    python onnx\backend\test\stat_coverage.py
    python onnx\defs\gen_doc.py
    set ONNX_ML=0
    python onnx\defs\gen_doc.py
    set ONNX_ML=1

Update an existing operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All operators are defined in folder
`onnx/onnx/defs <https://github.com/onnx/onnx/tree/master/onnx/defs>`_.
There are two files in every subfolder, one called `defs.cc` and another one
called `old.cc`.

* `defs.cc`: contains the most recent definition for every operator
* `old.cc`: contains the deprecated version of the operators in previous opset

Updating an operator means copying the definition from `defs.cc` to `old.cc`
and updating the existing one in `defs.cc`.

One file following the pattern `onnx/defs/operator_sets*.h`
must be modified. These headers registers the list
of existing operators.

File `onnx/defs/schema.h
<https://github.com/onnx/onnx/tree/master/onnx/defs/schema.h>`_
contains the latest opset version. It must updated too if one opset
was upgraded.

File `onnx/version_converter/convert.h
<https://github.com/onnx/onnx/tree/master/onnx/version_converter/convert.h>`_
contains rules to apply when converter a node from an opset to the next one.
This file may be updated too.

The package must be compiled and the documentation must be generated
again to automatically update the markdown documentation and it must
be included into the PR.

Then unit test must be updated.

**Summary**

* Modify files `defs.cc`, `old.cc`, `onnx/defs/operator_sets*.h`,
  `onnx/defs/schema.h`
* Optional: modify file `onnx/version_converter/convert.h`
* Build onnx.
* Build the documentation.
* Update unit test.

The PR should include the modified files and the modified markdown documentation,
usually a subset of
`docs/docs/Changelog-ml.md`, `docs/Changelog.md`,
`docs/Operators-ml.md`, `docs/Operators.md`,
`docs/TestCoverage-ml.md`, `docs/TestCoverage.md`.
