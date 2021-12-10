
============================
Build ONNX Graph with Python
============================

Next sections highlight the main function used to build
an ONNX graph.

.. contents::
    :local:

A simple example: a linear regression
=====================================

.. _l-onnx-linear-regression-onnx-api:

Linear Regression, no initializer
+++++++++++++++++++++++++++++++++

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
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

    # outputs
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)

    # nodes
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])

    # from nodes to graph
    graph = make_graph([node1, node2],  # nodes
                        'lr',  # a name
                        [X, A, B],  # inputs
                        [Y])  # outputs
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

    # inputs
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

    # outputs
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)

    # nodes
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])

    # from nodes to graph
    graph = make_graph([node1, node2],  # nodes
                        'lr',  # a name
                        [X, A, B],  # inputs
                        [Y])  # outputs
    onnx_model = make_model(graph)
    print(OnnxInference(onnx_model).to_dot())

.. _l-onnx-linear-regression-onnx-api-init:

Linear Regression, initializer
++++++++++++++++++++++++++++++

.. runpython::
    :showcode:
    :toggle: out
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

    # input
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])

    # output
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)

    # nodes
    node1 = make_node('MatMul', ['X', 'C'], ['AX'])
    node2 = make_node('Add', ['AX', 'C'], ['Y'])

    # graph
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

    # initializers
    value = numpy.array([0.5, -0.6], dtype=numpy.float32)
    A = numpy_helper.from_array(value, name='A')

    value = numpy.array([0.4], dtype=numpy.float32)
    C = numpy_helper.from_array(value, name='C')

    # input
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])

    # output
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)

    # nodes
    node1 = make_node('MatMul', ['X', 'C'], ['AX'])
    node2 = make_node('Add', ['AX', 'C'], ['Y'])

    # graph
    graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
    onnx_model = make_model(graph)
    print(OnnxInference(onnx_model).to_dot())

ONNX Python API
===============

make functions
++++++++++++++

Write, Read an ONNX graph
+++++++++++++++++++++++++

What is a converting library?
+++++++++++++++++++++++++++++

Other API
+++++++++
