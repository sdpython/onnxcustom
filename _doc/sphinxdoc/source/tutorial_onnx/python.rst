
============================
Build ONNX Graph with Python
============================

Next sections highlight the main functions used to build
an ONNX graph with the :ref:`Python API <l-python-onnx-api>`
:epkg:`onnx` offers.

.. contents::
    :local:

.. _l-onnx-linear-regression-onnx-api:

A simple example: a linear regression
=====================================

The linear regression is the most simple model
in machine learning described by the following expression
:math:`Y = XA + B`. We can see it as a function of three
variables :math:`Y = f(X, A, B)` decomposed into
`y = Add(MatMul(X, A), B))`. That what's we need to represent
with ONNX operators. The first is to implement a function
with :ref:`ONNX operators <l-onnx-operators>`.
ONNX is strongly typed. Shape and type must be defined for both
input and output of the function. That said, we need four functions
to build the graph among the :ref:`l-onnx-make-function`:

* `make_tensor_value_info`: declares a variable (input or output)
  given its shape and type
* `make_node`: creates a node defined by an operation
  (an operator type), its inputs and outputs
* `make_graph`: a function to create an ONNX graph with
  the objects created by the two previous functions
* `make_model`: a last function with merges the graph and
  additional metadata

All along the creation, we need to give a name to every input,
output of every node of the graph. Input and output of the graph
are defined by :epkg:`onnx` objects, strings are used to refer to
intermediate results. This is how it looks like.

.. runpython::
    :showcode:
    :toggle: out
    :warningout: DeprecationWarning

    # imports
    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor,
        make_graph, make_tensor_value_info)
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    # inputs

    # 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

    # outputs, the shape is left undefined

    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)

    # nodes

    # It creates a node defined by the operator type MatMul,
    # 'X', 'A' are the inputs of the node, 'XA' the output.
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])

    # from nodes to graph
    # the graph is built from the list of nodes, the list of inputs,
    # the list of outputs and a name.

    graph = make_graph([node1, node2],  # nodes
                        'lr',  # a name
                        [X, A, B],  # inputs
                        [Y])  # outputs

    # onnx graph
    # there is no metata in this case.

    onnx_model = make_model(graph)

    # the work is done, let's display it...
    print(onnx_simple_text_plot(onnx_model))

.. gdot::
    :script:

    from mlprodict.testing.einsum import decompose_einsum_equation
    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor,
        make_graph, make_tensor_value_info)
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])
    graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
    onnx_model = make_model(graph)
    print(OnnxInference(onnx_model).to_dot())

An empty shape (`None`) means any shape, a shapes defines as `[None, None]`
tells this object is a tensor with two dimensions without any further precision.
The ONNX graph can also be inspected by looking into the fields
of each object of the graph.

.. runpython::
    :showcode:

    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor,
        make_graph, make_tensor_value_info)
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    def shape2tuple(shape):
        return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])
    graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
    onnx_model = make_model(graph)

    # the list of inputs
    print(onnx_model.graph.input)

    # in a more nicely format
    for obj in onnx_model.graph.input:
        print("name=%r dtype=%r shape=%r" % (
            obj.name, obj.type.tensor_type.elem_type,
            shape2tuple(obj.type.tensor_type.shape)))

    # the list of outputs
    print(onnx_model.graph.output)

    # in a more nicely format
    for obj in onnx_model.graph.output:
        print("name=%r dtype=%r shape=%r" % (
            obj.name, obj.type.tensor_type.elem_type,
            shape2tuple(obj.type.tensor_type.shape)))

    # the list of nodes
    print(onnx_model.graph.output)

    # in a more nicely format
    for node in onnx_model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))

The tensor type is an integer. The following array gives the
equivalent type with :epkg:`numpy`.

.. runpython::
    :showcode:

    import pprint
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

    pprint.pprint(TENSOR_TYPE_TO_NP_TYPE)

Serialization
=============

ONNX is based on :epkg:`protobuf`. It minimizes the space needed
to save the graph on disk. Every object (see :ref:`l-onnx-classes`)
in :epkg:`onnx` can be serialized with method `SerializeToString`. That's
the case for the whole model.

.. runpython::
    :showcode:

    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor,
        make_graph, make_tensor_value_info)
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    def shape2tuple(shape):
        return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])
    graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
    onnx_model = make_model(graph)

    # The serialization
    with open("linear_regression.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # display
    print(onnx_simple_text_plot(onnx_model))

The graph can be restored with function `load`:

.. runpython::
    :showcode:

    from onnx import load
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    with open("linear_regression.onnx", "rb") as f:
        onnx_model = load(f)

    # display
    print(onnx_simple_text_plot(onnx_model))

It looks exactly the same. Any model can be serialized this way
unless they are bigger than 2 Gb. :epkg:`protobuf` is limited to size
smaller than this threshold. Next section show how to overcome that limit.

.. _l-onnx-linear-regression-onnx-api-init:

Initializer, default value
++++++++++++++++++++++++++

The previous model assumed the coefficients of the linear regression
were also input of the model. That's not very convenient. They should be
part of the model itself as constant or **initializer** to follow
onnx semantic. Next example modifies the previous one to change inputs
`A` and `B` into initializer. The package implements two functions to
convert from :epkg:`numpy` into :epkg:`onnx` and the other way around
(see :ref:`l-numpy-helper-onnx-array`).

* `onnx.numpy_helper.to_array`: converts from onnx to numpy
* `onnx.numpy_helper.from_array`: converts from numpy to onnx

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor, make_graph,
        make_tensor_value_info)
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    # initializers
    value = numpy.array([0.5, -0.6], dtype=numpy.float32)
    A = numpy_helper.from_array(value, name='A')

    value = numpy.array([0.4], dtype=numpy.float32)
    C = numpy_helper.from_array(value, name='C')

    # the part which does not change
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
    node1 = make_node('MatMul', ['X', 'C'], ['AX'])
    node2 = make_node('Add', ['AX', 'C'], ['Y'])
    graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
    onnx_model = make_model(graph)

    print(onnx_simple_text_plot(onnx_model))

.. gdot::
    :script:

    from mlprodict.testing.einsum import decompose_einsum_equation
    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor,
        make_graph, make_tensor_value_info)
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    value = numpy.array([0.5, -0.6], dtype=numpy.float32)
    A = numpy_helper.from_array(value, name='A')
    value = numpy.array([0.4], dtype=numpy.float32)
    C = numpy_helper.from_array(value, name='C')
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
    node1 = make_node('MatMul', ['X', 'C'], ['AX'])
    node2 = make_node('Add', ['AX', 'C'], ['Y'])
    graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
    onnx_model = make_model(graph)
    print(OnnxInference(onnx_model).to_dot())

Again, it is possible to go through the onnx structure to check
how the initializer look like.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor, make_graph,
        make_tensor_value_info)
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    # initializers
    value = numpy.array([0.5, -0.6], dtype=numpy.float32)
    A = numpy_helper.from_array(value, name='A')

    value = numpy.array([0.4], dtype=numpy.float32)
    C = numpy_helper.from_array(value, name='C')

    # the part which does not change
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
    node1 = make_node('MatMul', ['X', 'C'], ['AX'])
    node2 = make_node('Add', ['AX', 'C'], ['Y'])
    graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
    onnx_model = make_model(graph)

    for init in onnx_model.graph.initializer:
        print(init)

The type is defined as integer as well with the same meaning.
In this second example, there is only one input left.
Input `A` and `B` were removed. They could be kept. In that case,
they are optional. The user can compute the predictions. Every undefined
input is replaced by the corresponding initializer. It is a default value.

Attributes
==========

Some operators needs attributes such as :epkg:`Transpose` operator.
Let's build the graph for expression :epkg:`y = XA' + B` or
`y = Add(MatMul(X, Transpose(A)) + B)`. Tranpose needs an attribute
defining the permutation of axes: `perm=[1, 0]`. It is added
as a named attribute in function `make_node`.

.. runpython::
    :showcode:

    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor,
        make_graph, make_tensor_value_info)
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    # unchanged
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)

    # added
    node_transpose = make_node('Transpose', ['A'], ['tA'], perm=[1, 0])

    # unchanged except A is replaced by tA
    node1 = make_node('MatMul', ['X', 'tA'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])

    # node_transpose is added to the list
    graph = make_graph([node_transpose, node1, node2],
                       'lr', [X, A, B], [Y])
    onnx_model = make_model(graph)

    # the work is done, let's display it...
    print(onnx_simple_text_plot(onnx_model))

.. gdot::
    :script:

    from mlprodict.testing.einsum import decompose_einsum_equation
    import numpy
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model, make_node, set_model_props, make_tensor,
        make_graph, make_tensor_value_info)
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
    node_transpose = make_node('Transpose', ['A'], ['tA'], perm=[1, 0])
    node1 = make_node('MatMul', ['X', 'tA'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])
    graph = make_graph([node_transpose, node1, node2],
                       'lr', [X, A, B], [Y])
    onnx_model = make_model(graph)
    print(OnnxInference(onnx_model).to_dot())

Opset and metadata
==================

Let's load the ONNX file previously created and check
what kind of metadata it has.

.. runpython::
    :showcode:

    from onnx import load

    with open("linear_regression.onnx", "rb") as f:
        onnx_model = load(f)

    for field in ['doc_string', 'domain', 'functions',
                  'ir_version', 'metadata_props', 'model_version',
                  'opset_import', 'producer_name', 'producer_version',
                  'training_info']:
        print(field, getattr(onnx_model, field))

Most of them are empty because it was not filled when the ONNX
graph was created. Two of them have a value:

.. runpython::
    :showcode:

    from onnx import load

    with open("linear_regression.onnx", "rb") as f:
        onnx_model = load(f)

    print("ir_version:", onnx_model.ir_version)
    for opset in onnx_model.opset_import:
        print("opset domain=%r version=%r" % (opset.domain, opset.version))

:epkg:`IR` defined the version of ONNX language version.
Opset defines the version of operators being used.
Without any precision, ONNX uses the latest version available
coming from the installed package.
Another one can be used.

.. runpython::
    :showcode:

    from onnx import load

    with open("linear_regression.onnx", "rb") as f:
        onnx_model = load(f)

    del onnx_model.opset_import[:]
    opset = onnx_model.opset_import.add()
    opset.domain = ''
    opset.version = 14

    for opset in onnx_model.opset_import:
        print("opset domain=%r version=%r" % (opset.domain, opset.version))

Any opset can be used as long as all operators are defined
the way ONNX specifies it. Version 5 of operator *Reshape*
defines the shape as an input and not as an attribute like in
version 1. The opset tells which specifications is followed
while describing the graph.

The other metadata can be used to store any information,
to store information about the way the model was generated,
a way to distinguish a model from another one with a version
number.

.. runpython::
    :showcode:

    from onnx import load, helper, TrainingInfoProto

    with open("linear_regression.onnx", "rb") as f:
        onnx_model = load(f)

    onnx_model.model_version = 15
    onnx_model.producer_name = "something"
    onnx_model.producer_version = "some other thing"
    onnx_model.doc_string = "documentation about this model"
    prop = onnx_model.metadata_props

    data = dict(key1="value1", key2="value2")
    helper.set_model_props(onnx_model, data)

    print(onnx_model)

Field `training_info` can be used to store additional graphs.
See `training_tool_test.py
<https://github.com/onnx/onnx/blob/master/onnx/test/training_tool_test.py>`_
to see how it works.

Subgraph: If, Scan, Loop
========================

Parsing
=======

Module :epkg:`onnx` provides a faster way to define a graph
a lot easier to read. That's easy to use when the graph is built
in a single function, less easy when the graph is built from many
different functions converting each piece of a machine learning
pipeline.

.. runpython::
    :showcode:

    import onnx.parser
    from mlprodict.plotting.text_plot import onnx_simple_text_plot

    input = '''
        <
            ir_version: 8,
            opset_import: [ "" : 15]
        >
        agraph (float[I,J] X, float[] A, float[] B) => (float[I] Y) {
            XA = MatMul(X, A)
            Y = Add(XA, B)
        }
        '''
    onnx_model = onnx.parser.parse_model(input)

    print(onnx_simple_text_plot(onnx_model))

What is a converting library?
+++++++++++++++++++++++++++++

Other API
+++++++++
