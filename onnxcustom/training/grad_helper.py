"""
@file
@brief ONNX and gradient.
"""
from io import BytesIO
import onnx
from onnx.helper import make_model, make_graph
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtModuleGraphBuilder,
    OrtModuleGraphBuilderConfiguration,
    TrainingGraphTransformerConfiguration)
from ..utils.orttraining_helper import get_train_initializer


def onnx_derivative(onx, weights=None, inputs=None,
                    out_yield_op=True, keep_output=True):
    """
    Builds the gradient for an onnx graph.

    :param onx: onnx graph
    :param weights: gradient against those weights, None for all real weights
    :param inputs: gradient against inputs, None for all real inputs
    :param out_yield_op: promotes yield operator as output
    :param keep_output: keep the function output or only returns the
        gradient, this parameter is unused if *out_yield_op* is False
    :return: onnx graph

    The function calls :epkg:`OrtModuleGraphBuilderConfiguration`
    from :epkg:`onnxruntime-training`. This graph is meant to be used
    with :epkg:`OrtGradientForwardBackward` and includes
    operator `YieldOp`. That's the graph looks this way:

    .. gdot::
        :script: DOT-SECTION

        import numpy
        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxAdd, OnnxMul, OnnxIdentity)
        from skl2onnx.common.data_types import FloatTensorType
        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.training.grad_helper import onnx_derivative
        from onnxcustom import __max_supported_opset__ as opv

        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, out_yield_op=False)

        oinf = OnnxInference(new_onx)
        print("DOT-SECTION", oinf.to_dot())

    These operators are the outputs of the
    initial graph and must be replaced by the gradient of these
    outputs to compute the gradient of the weights and the inputs.
    After they are replaced, it looks this way:

    .. gdot::
        :script: DOT-SECTION

        import numpy
        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxAdd, OnnxMul, OnnxIdentity)
        from skl2onnx.common.data_types import FloatTensorType
        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.training.grad_helper import onnx_derivative
        from onnxcustom import __max_supported_opset__ as opv

        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, out_yield_op=True, keep_output=False)

        oinf = OnnxInference(new_onx)
        print("DOT-SECTION", oinf.to_dot())

    The user can still compute the outputs.

    .. gdot::
        :script: DOT-SECTION

        import numpy
        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxAdd, OnnxMul, OnnxIdentity)
        from skl2onnx.common.data_types import FloatTensorType
        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.training.grad_helper import onnx_derivative
        from onnxcustom import __max_supported_opset__ as opv

        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, out_yield_op=True, keep_output=True)

        oinf = OnnxInference(new_onx)
        print("DOT-SECTION", oinf.to_dot())
    """
    if weights is None:
        inits = get_train_initializer(onx)
        weights = list(inits)
    builder = OrtModuleGraphBuilder()

    config = OrtModuleGraphBuilderConfiguration()
    config.initializer_names = weights
    config.initializer_names_to_train = weights
    if inputs is None:
        inputs_name = []
        for i in onx.graph.input:
            try:
                elem_type = i.type.tensor_type.elem_type
            except AttributeError:
                # not a vector
                continue
            if elem_type in (
                    onnx.TensorProto.FLOAT16,  # pylint: disable=E1101
                    onnx.TensorProto.FLOAT,  # pylint: disable=E1101
                    onnx.TensorProto.DOUBLE):  # pylint: disable=E1101
                inputs_name.append(i.name)
        if len(inputs_name) > 0:
            config.input_names_require_grad = inputs_name
    config.build_gradient_graph = True

    p = TrainingGraphTransformerConfiguration()
    config.graph_transformer_config = p

    builder.initialize(onx.SerializeToString(), config)
    builder.build()
    train_onnx_model_serialized = builder.get_model()
    # optimized_pre_grad_model = builder.get_inference_optimized_model()
    grad_yield = onnx.load(BytesIO(train_onnx_model_serialized))
    if not out_yield_op:
        return grad_yield

    yields_op = [
        node for node in grad_yield.graph.node  # pylint: disable=E1101
        if node.op_type == 'YieldOp']
    if len(yields_op) == 0:
        raise RuntimeError(  # pragma: no cover
            "No YieldOp was found. The input graph must be wrong.")

    other_nodes = [
        node for node in grad_yield.graph.node  # pylint: disable=E1101
        if node.op_type != 'YieldOp']
    inputs = list(grad_yield.graph.input)  # pylint: disable=E1101
    outputs = list(grad_yield.graph.output)  # pylint: disable=E1101
    map_out = {o.name: o for o in onx.graph.output}
    for yn in yields_op:
        if len(yn.input) != 1 or len(yn.output) != 1:
            raise NotImplementedError(  # pragma: no cover
                "Unexpected configuration for YieldOp node %r." % yn)
        if yn.input[0] not in map_out:
            raise RuntimeError(  # pragma: no cover
                "Unable to find output %r in %r." % (
                    yn.input[0], list(map_out)))
        out = map_out[yn.input[0]]
        new_input = onnx.ValueInfoProto()
        new_input.name = yn.output[0]
        new_input.doc_string = "from yieldop"
        new_input.type.CopyFrom(out.type)  # pylint: disable=E1101
        inputs.append(new_input)

        if keep_output:
            # Keeps output from the original graph.
            outputs.append(out)

    # Final graph.
    graph = make_graph(
        other_nodes, grad_yield.graph.name, inputs, outputs,  # pylint: disable=E1101
        list(grad_yield.graph.initializer))  # pylint: disable=E1101
    new_model = make_model(graph)
    new_model.ir_version = grad_yield.ir_version  # pylint: disable=E1101
    new_model.producer_name = grad_yield.producer_name  # pylint: disable=E1101
    new_model.producer_version = grad_yield.producer_version  # pylint: disable=E1101
    new_model.domain = grad_yield.domain  # pylint: disable=E1101
    new_model.model_version = grad_yield.model_version  # pylint: disable=E1101
    new_model.doc_string = grad_yield.doc_string  # pylint: disable=E1101
    if hasattr(onx, 'value_info'):
        graph.value_info.extend(grad_yield.value_info)  # pylint: disable=E1101
    del new_model.opset_import[:]  # pylint: disable=E1101
    for oimp in grad_yield.opset_import:  # pylint: disable=E1101
        op_set = new_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    return new_model
