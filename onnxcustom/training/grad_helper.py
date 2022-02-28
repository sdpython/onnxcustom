# pylint: disable=E1101
"""
@file
@brief ONNX and gradient.
"""
from io import BytesIO
from enum import IntFlag
import onnx
from onnx.helper import make_model, make_graph, make_node, make_tensor
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtModuleGraphBuilder,
    OrtModuleGraphBuilderConfiguration,
    TrainingGraphTransformerConfiguration)
from mlprodict.onnx_tools.optim.onnx_optimisation import onnx_remove_node
from ..utils.orttraining_helper import get_train_initializer


class DerivativeOptions(IntFlag):
    """
    Options defining how to build the onnx graph of the
    gradients.

    * `Zero`: default option, all options are disabled
    * `KeepYieldOp`: keeps the operator *YieldOp* in the graph,
      see @see fn onnx_derivative
    * `KeepOutputs`: keeps the output of the original graph
    * `FillGrad`: does not add any output to specify the gradient
      of the output but assumes it is one
    * `Loss`: the function assumes the loss was added to the graph
    """

    Zero = 0
    KeepYieldOp = 1
    KeepOutputs = 2
    FillGrad = 4
    Loss = 5


def onnx_derivative(onx, weights=None, inputs=None,
                    options=DerivativeOptions.Zero,
                    loss=None, label=None, path_name=None):
    """
    Builds the gradient for an onnx graph.

    :param onx: onnx graph
    :param weights: gradient against those weights, None for all real weights
    :param inputs: gradient against inputs, None for all real inputs
    :param options: options of type @see cl DerivativeOptions
    :param loss: loss output in case a loss was added in the graph,
        *options* must be equal to `DerivativeOptions.Loss`
    :param label: if *loss* is specified, then the label must be
        specified as well
    :param path_name: if *options* equal to `DerivativeOptions.Loss`,
        the gradient is saved to that path
    :return: onnx graph

    The function calls :epkg:`OrtModuleGraphBuilderConfiguration`
    from :epkg:`onnxruntime-training`. This graph is meant to be used
    with @see cl OrtGradientForwardBackward and includes
    operator `YieldOp`. That's the graph looks this way:

    .. gdot::
        :script: DOT-SECTION

        import numpy
        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxAdd, OnnxMul, OnnxIdentity)
        from skl2onnx.common.data_types import FloatTensorType
        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.training.grad_helper import (
            onnx_derivative, DerivativeOptions)
        from onnxcustom import __max_supported_opset__ as opv

        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, options=DerivativeOptions.KeepYieldOp)

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
        from onnxcustom.training.grad_helper import (
            onnx_derivative, DerivativeOptions)
        from onnxcustom import __max_supported_opset__ as opv

        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, options=DerivativeOptions.Zero)

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
        from onnxcustom.training.grad_helper import (
            onnx_derivative, DerivativeOptions)
        from onnxcustom import __max_supported_opset__ as opv

        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, options=DerivativeOptions.KeepOutputs)

        oinf = OnnxInference(new_onx)
        print("DOT-SECTION", oinf.to_dot())

    The input gradient can be filled with a constant matrix
    filled with one and with the expected shape.

    .. gdot::
        :script: DOT-SECTION

        import numpy
        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxAdd, OnnxMul, OnnxIdentity)
        from skl2onnx.common.data_types import FloatTensorType
        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.training.grad_helper import (
            onnx_derivative, DerivativeOptions)
        from onnxcustom import __max_supported_opset__ as opv

        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, 10])},
                           {'Y': FloatTensorType([None, 10])},
                           target_opset=opv)
        new_onx = onnx_derivative(onx, options=(
            DerivativeOptions.KeepOutputs | DerivativeOptions.FillGrad))

        oinf = OnnxInference(new_onx)
        print("DOT-SECTION", oinf.to_dot())
    """
    if not isinstance(options, DerivativeOptions):
        raise TypeError(
            "Options must be from type DerivativeOptions not %r."
            "" % type(options))

    if options == DerivativeOptions.Loss:
        return _onnx_derivative_loss(onx, weights=weights, inputs=inputs,
                                     options=options, loss=loss, label=label,
                                     path_name=path_name)
    return _onnx_derivative_fw(onx, weights=weights, inputs=inputs,
                               options=options)


def _default_inputs(onx):
    "Guesses default inputs (float ones) if not specified."
    inputs_name = []
    for i in onx.graph.input:
        try:
            elem_type = i.type.tensor_type.elem_type
        except AttributeError:  # pragma: no cover
            # not a vector
            continue
        if elem_type in (
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.DOUBLE):
            inputs_name.append(i.name)
    return inputs_name


def _onnx_derivative_fw(onx, weights, inputs, options):
    """
    Implements a gradient based on class `OrtModuleGraphBuilder`.
    """
    if weights is None:
        inits = get_train_initializer(onx)
        weights = list(inits)
    builder = OrtModuleGraphBuilder()

    config = OrtModuleGraphBuilderConfiguration()
    config.initializer_names = weights
    config.initializer_names_to_train = weights
    if inputs is None:
        inputs_name = _default_inputs(onx)
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
    if options & DerivativeOptions.KeepYieldOp:
        if options != DerivativeOptions.KeepYieldOp:
            raise ValueError(
                "Option YieldOd cannot be combined with any other.")
        return grad_yield

    yields_op = [
        node for node in grad_yield.graph.node
        if node.op_type == 'YieldOp']
    if len(yields_op) == 0:
        raise RuntimeError(  # pragma: no cover
            "No YieldOp was found. The input graph must be wrong.")

    other_nodes = [
        node for node in grad_yield.graph.node
        if node.op_type != 'YieldOp']
    inputs = list(grad_yield.graph.input)
    if options & DerivativeOptions.KeepOutputs:
        outputs = list(grad_yield.graph.output)
    else:
        original = set(i.name for i in onx.graph.output)
        outputs = [o for o in grad_yield.graph.output
                   if o.name not in original]
    map_out = {o.name: o for o in onx.graph.output}
    for yn in yields_op:
        if len(yn.input) != 1 or len(yn.output) != 1:
            raise NotImplementedError(  # pragma: no cover
                "Unexpected configuration for YieldOp node %r." % yn)
        if yn.input[0] not in map_out:
            raise RuntimeError(  # pragma: no cover
                "Unable to find output %r in %r." % (
                    yn.input[0], list(map_out)))
        if not(options & DerivativeOptions.FillGrad):  # pylint: disable=C0325
            out = map_out[yn.input[0]]
            new_input = onnx.ValueInfoProto()
            new_input.name = yn.output[0]
            new_input.doc_string = "from yieldop"
            new_input.type.CopyFrom(out.type)
            inputs.append(new_input)
        else:
            if not(options & DerivativeOptions.KeepOutputs):  # pylint: disable=C0325
                raise ValueError(  # pragma: no cover
                    "FillGrad should be set with KeepOutputs.")
            name = "%s_shape" % yn.input[0]
            node = make_node('Shape', [yn.input[0]], [name])
            other_nodes.append(node)
            out = map_out[yn.input[0]]
            elem_type = out.type.tensor_type.elem_type
            node = make_node(
                'ConstantOfShape', [name], [yn.output[0]],
                value=make_tensor(
                    "value", elem_type, (1, ), [1]))
            other_nodes.append(node)
        if options & DerivativeOptions.KeepOutputs:
            # Keeps output from the original graph.
            outputs.append(out)

    # Final graph.
    graph = make_graph(
        other_nodes, grad_yield.graph.name, inputs, outputs,
        list(grad_yield.graph.initializer))
    new_model = make_model(graph)
    new_model.ir_version = grad_yield.ir_version
    new_model.producer_name = grad_yield.producer_name
    new_model.producer_version = grad_yield.producer_version
    new_model.domain = grad_yield.domain
    new_model.model_version = grad_yield.model_version
    new_model.doc_string = grad_yield.doc_string
    if hasattr(onx, 'value_info'):
        graph.value_info.extend(grad_yield.value_info)
    del new_model.opset_import[:]
    for oimp in grad_yield.opset_import:
        op_set = new_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    return onnx_remove_node(new_model)


def _onnx_derivative_loss(onx, weights, inputs, options, loss, label,
                          path_name):
    """
    Implements a gradient based on class `PyGradientGraphBuilder`.
    """
    from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
        GradientGraphBuilder)
    if path_name is None:
        raise ValueError(
            "path_name must not be None if options is 'Loss'.")
    if weights is not None:
        raise ValueError(
            "weights must be None if options is 'Loss'.")
    if label is None:
        raise ValueError(
            "label must not be None if options is 'Loss'.")
    if loss is None or not isinstance(loss, str):
        raise ValueError(
            "loss must not None and a string if options is 'Loss'.")
    if isinstance(label, str):
        label = {label}
    else:
        label = set(label)
    if inputs is None:
        inputs_name = _default_inputs(onx)
        inputs = inputs_name
    if isinstance(inputs, str):
        inputs = {inputs}
    else:
        inputs = set(inputs)
    inputs = set(x for x in inputs if x not in label)

    builder = GradientGraphBuilder(
        onx.SerializeToString(), label, inputs, loss)
    builder.build()
    builder.save(path_name)
    with open(path_name, "rb") as f:
        return onnx.load(f)
