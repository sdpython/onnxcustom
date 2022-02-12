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


def onnx_derivative(onx, weights=None, include_inputs=True):
    """
    Builds the gradient for an onnx graph.

    :param onx: onnx graph
    :param weights: gradient against those weights
    :param include_inputs: include inputs
    :return: onnx graph
    """
    if weights is None:
        inits = get_train_initializer(onx)
        weights = list(inits)
    builder = OrtModuleGraphBuilder()

    config = OrtModuleGraphBuilderConfiguration()
    config.initializer_names = weights
    config.initializer_names_to_train = weights
    if include_inputs:
        input_names = [n.name for n in onx.graph.input]
        config.input_names_require_grad = input_names
    config.build_gradient_graph = True

    p = TrainingGraphTransformerConfiguration()
    config.graph_transformer_config = p

    builder.initialize(onx.SerializeToString(), config)
    builder.build()
    train_onnx_model_serialized = builder.get_model()
    # optimized_pre_grad_model = builder.get_inference_optimized_model()
    return onnx.load(BytesIO(train_onnx_model_serialized))
