"""
@file
@brief ONNX and gradient.
"""
from io import BytesIO
import onnx
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

    yields_op = [node for node in grad_yield.graph.node  # pylint: disable=E1101
                 if node.op_type == 'YieldOp']
    if len(yields_op) == 0:
        raise RuntimeError(  # pragma: no cover
            "No YieldOp was found. The input graph must be wrong.")
