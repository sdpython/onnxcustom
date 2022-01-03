# pylint: disable=E1101
"""
@file
@brief Gradient with :epkg:`onnxruntime-training` forward backward.
"""
import logging
from io import BytesIO
import onnx
from onnx.numpy_helper import to_array
from onnxruntime import InferenceSession, RunOptions
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    SessionIOBinding, OrtValue as C_OrtValue)
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    TrainingAgent, OrtValueCache, OrtModuleGraphBuilder,
    OrtModuleGraphBuilderConfiguration, OrtDevice,
    TrainingGraphTransformerConfiguration, OrtValueVector,
    PartialGraphExecutionState)
from ..utils.orttraining_helper import get_train_initializer


class OrtGradientForwardBackward:
    """
    Implements forward backward mechanism assuming the function
    to train is defined by an ONNX graph.

    :param onnx_model: onnx model
    :param weights_to_train: names of the weights to train,
        if None, all initializer of floats type are included in the list
    :param input_names: input names or None for all
    :param output_names: output names or None for all
    :param class_name: name to give the class dynamically created
    :param sess_options: see :epkg:`SessionOptions`
    :param providers: see :epkg:`InferenceSession`
    :param provider_options: see :epkg:`InferenceSession`
    :param run_options: see :epkg:`RunOptions`
    :param graph_builder_config:
        see :epkg:`OrtModuleGraphBuilderConfiguration`
    :param device_index: used for cuda (0 for `cuda:0`,
        `cuda:1`, ...), 0 by default
    :param enable_logging: enables logging while setting up the class
    :param debug: to run extra verification while training

    .. note::
        The current implementation of :epkg:`onnxruntime` forces
        the weights to train to appear in the alphabetical order.
        The constructor checks that condition is verified.

    .. warning::
        This class does not consider subgraphs.
    """

    def __init__(self, onnx_model, weights_to_train=None,
                 input_names=None, output_names=None, class_name=None,
                 sess_options=None, providers=None,
                 provider_options=None, run_options=None,
                 graph_builder_config=None,
                 device_index=0, enable_logging=False, debug=False):

        if weights_to_train is None:
            weights_to_train = (
                OrtGradientForwardBackward._select_initializer_names(
                    onnx_model))
            if len(weights_to_train) == 0:
                raise RuntimeError(  # pragma: no cover
                    "Unable to guess the weights to train from initializers: "
                    "%r." % [i.name for i in onnx_model.graph.initializer])

        self.onnx_model = onnx_model
        self.input_names = input_names
        self.output_names = output_names
        self.weights_to_train = weights_to_train
        self.device_index = device_index
        self.enable_logging = enable_logging
        self.class_name = (class_name if class_name is not None else
                           "OrtGradientForwardBackwardFunction_%d" % id(self))

        self.provider_options = provider_options
        self.sess_options = sess_options
        self.providers = providers
        self.run_options = run_options
        self.graph_builder_config = graph_builder_config
        self.debug = debug

        # default
        if self.weights_to_train is None:
            raise ValueError(  # pragma: no cover
                "weights_to_train must be specified.")
        if self.input_names is None:
            self.input_names = [obj.name
                                for obj in self.onnx_model.graph.input]
        if self.output_names is None:
            self.output_names = [obj.name
                                 for obj in self.onnx_model.graph.output]
        if self.class_name is None:
            self.class_name = "TorchOrtFunction_%r" % id(
                self)  # pragma: no cover
        if hasattr(self.providers, 'type'):
            if self.providers.type != 'cpu':
                self.device_index = self.providers.index
            self.providers = self.providers.type
        if self.providers in (None, 'cpu'):
            self.providers = ["CPUExecutionProvider" for i in self.input_names]
            if self.provider_options is None:
                self.provider_options = [{} for i in self.input_names]
        elif self.providers in ('cuda', 'cuda:0', 'gpu'):
            self.providers = [
                "CUDAExecutionProvider" for i in self.input_names]
            if self.provider_options is None:
                self.provider_options = [{} for i in self.input_names]
        if len(self.input_names) != len(self.providers):
            raise ValueError(  # pragma: no cover
                "input_names and providers must have the same length.")
        if self.provider_options is None:
            self.provider_options = [{} for i in self.input_names]
        if len(self.input_names) != len(self.provider_options):
            raise ValueError(  # pragma: no cover
                "input_names and provider_options must have the same length.")

        if list(sorted(self.weights_to_train)) != self.weights_to_train:
            raise ValueError(  # pragma: no cover
                "List of weights to train must be sorted but %r is not. "
                "You shoud use function onnx_rename_weights to do that "
                "before calling this class." % self.weights_to_train)
        set_weights = set(self.weights_to_train)
        found = []
        for i in self.onnx_model.graph.initializer:
            if i.name not in set_weights:
                continue
            found.append(i.name)
        if len(found) != len(self.weights_to_train):
            raise ValueError(
                "One weight name in self.weights_to_train was not found in "
                "the initializers %r." % (self.weights_to_train, ))
        if found != self.weights_to_train:
            raise ValueError(
                "List of weights to train must be sorted and follow the "
                "as the initializers in the graph. %r != %r."
                "You shoud use function onnx_rename_weights to do that "
                "before calling this class." % (
                    self.weights_to_train, found))

        if any(map(lambda v: v not in ['CPUExecutionProvider',
                                       'CUDAExecutionProvider'],
                   self.providers)):
            raise ValueError(
                "Unexpected providers %r (providers=%r)." % (
                    self.providers, providers))

        # complete initialisation
        self._init_next()

    @staticmethod
    def _select_initializer_names(onnx_model):
        """
        Selects all initializers with float type.

        :param onnx_model: ONNX graph
        """
        inits = get_train_initializer(onnx_model)
        return list(inits)

    def _init_next(self):
        if self.enable_logging:
            self._logger = logging.getLogger("onnxcustom")
        else:
            self._logger = None  # pragma: no cover
        if self.run_options is None:
            self.run_options = RunOptions()
            self.run_options.training_mode = True

        if self.graph_builder_config is None:
            initializer_names = [
                i.name for i in self.onnx_model.graph.initializer]
            input_names = [i.name for i in self.onnx_model.graph.input]

            config = OrtModuleGraphBuilderConfiguration()
            config.initializer_names = [init for init in initializer_names
                                        if init in self.weights_to_train]
            config.initializer_names_to_train = self.weights_to_train
            config.input_names_require_grad = input_names
            config.build_gradient_graph = True

            if (len(config.initializer_names) !=  # noqa
                    len(config.initializer_names_to_train)):
                raise RuntimeError(  # pragma: no cover
                    "Unable to automatically fill "
                    "OrtModuleGraphBuilderConfiguration, mismatch between "
                    "%r and %r (initializer_names=%r)." % (
                        config.initializer_names,
                        config.initializer_names_to_train,
                        initializer_names))

            p = TrainingGraphTransformerConfiguration()
            config.graph_transformer_config = p

            # config.enable_caching = True
            # config.loglevel =
            # config.use_memory_efficient_gradient = True
            self.graph_builder_config = config

        attributes = self._create_onnx_graphs()
        attributes['__doc__'] = (
            "Inherits from @see cl OrtGradientForwardBackwardFunction.")
        attributes['__module__'] = (
            OrtGradientForwardBackwardFunction.__module__)
        self.cls_type_ = type(
            self.class_name, (OrtGradientForwardBackwardFunction,),
            attributes)

    def new_instance(self):
        """
        Creates an instance of class `self.cls_type_`.
        It implements methods *forward* and *backward*.
        """
        return self.cls_type_()

    def __getstate__(self):
        "Removes any non pickable attribute."
        atts = [k for k in self.__dict__ if not k.endswith('_')
                if k not in {'_logger', 'graph_builder_config',
                             'run_options'}]
        state = {att: getattr(self, att) for att in atts}
        state['run_options'] = None
        state['graph_builder_config'] = None
        return state

    def __setstate__(self, state):
        "Restores any non pickable attribute."
        for att, v in state.items():
            setattr(self, att, v)
        self._init_next()
        return self

    def __repr__(self):
        "usual"
        return "%s(...)" % self.__class__.__name__

    @staticmethod
    def _repr_helper_(obj, indent=0):
        "used to improve logging messages"
        if obj is None:
            return 'None'
        rows = []
        for c in sorted(dir(obj)):
            if c[0] == '_':
                continue
            try:
                value = getattr(obj, c)
            except AttributeError:  # pragma: no cover
                continue
            rows.append("%s=%r" % (c, value))

        if indent == 0:
            return "%s(%s)" % (obj.__class__.__name__, ", ".join(rows))
        return "%s(\n    %s)" % (
            obj.__class__.__name__,
            "\n    ".join(rows))

    @staticmethod
    def _provider_name_to_device_type(provider_name):
        if provider_name == 'CPUExecutionProvider':
            return OrtDevice.cpu()
        if provider_name == 'CUDAExecutionProvider':  # pragma: no cover
            return OrtDevice.cuda()
        raise ValueError(  # pragma: no cover
            'Unexpected provider name %r.' % provider_name)

    def get_initializer(self, name, exc=True):
        """
        Returns an initializer as numpy arrays.

        :param name: initializer name
        :param exc: raises an exception if not found or return None
        :return: the initializer as a :epkg:`C_OrtValue`
        """
        for init in self.onnx_model.graph.initializer:
            if name == init.name:
                return to_array(init)
        if exc:
            raise RuntimeError(  # pragma: no cover
                "Unable to find name %r in %r." % (
                    name,
                    list(i.name for i in self.onnx_model.graph.initializer)))
        return None

    def _create_onnx_graphs(self):
        """
        Creates forward and backward ONNX graph.
        The new class has the following attributes:

        * `__doc__`: doc string
        * `__module__`: module name (this file)
        * `_run_options`: see :epkg:`RunOptions`
        * `_sess`: :epkg:`InferenceSession` with the original graph
        * `_sess_eval`: :epkg:`InferenceSession` on the graph
            with weights as inputs
        * `_training_agent`: :epkg:`TrainingAgent`
        * `_cache`: :epkg:`OrtValueCache`
        * `_logger`: logger
        * `_input_names`: input names
        * `_debug`: use debug mode
        * `_grad_input_names`: gradient input names
        * `_output_names`: output names
        * `_weights_to_train`: names of the weights to train

        Training attributes

        * `_bw_fetches_names`: bw_fetches_names,
        * `_fw_outputs_device_info`: fw_outputs_device_info,
        * `_bw_outputs_device_info`: bw_outputs_device_info,
        * `_fw_no_grad_output_device_info`: fw_no_grad_output_device_info,
        * `_graph_info`: graph_info}

        Additional attributes added if *keep_model* is True:

        * `_trained_onnx`: ONNX graph for the gradient
        * `_optimized_pre_grad_model`: evaluation ONNX graph taking
            weights as inputs
        * `_graph_builder`: :epkg:`OrtModuleGraphBuilder`
        """
        logger = self._logger
        if logger is not None:
            logger.info("[OrtGradientForwardBackward] create training onnx")
            logger.info("[OrtGradientForwardBackward] input_names=%r",
                        self.input_names)
            logger.info("[OrtGradientForwardBackward] output_names=%r",
                        self.output_names)
            logger.info("[OrtGradientForwardBackward] weights_to_train=%r",
                        self.weights_to_train)

        builder = OrtModuleGraphBuilder()

        if logger is not None:
            cf = self.graph_builder_config.graph_transformer_config
            cfp = cf.propagate_cast_ops_config
            logger.info(
                "[OrtGradientForwardBackward] "
                "OrtModuleGraphBuilder.initialize")
            logger.info(
                "[OrtGradientForwardBackward] graph_builder_config=%s",
                OrtGradientForwardBackward._repr_helper_(
                    self.graph_builder_config, indent=4))
            logger.info(
                "[OrtGradientForwardBackward] graph_builder_config."
                "graph_transformer_config=%s",
                OrtGradientForwardBackward._repr_helper_(cf, indent=4))
            logger.info(
                "[OrtGradientForwardBackward] graph_builder_config."
                "graph_transformer_config.propagate_cast_ops_config=%s",
                OrtGradientForwardBackward._repr_helper_(cfp, indent=4))

        builder.initialize(
            self.onnx_model.SerializeToString(),
            self.graph_builder_config)

        if logger is not None:
            logger.info(
                "[OrtGradientForwardBackward] OrtModuleGraphBuilder.build")
        builder.build()

        if logger is not None:
            logger.info(
                "[OrtGradientForwardBackward] OrtModuleGraphBuilder.get_model")

        train_onnx_model_serialized = builder.get_model()

        optimized_pre_grad_model = builder.get_inference_optimized_model()
        graph_info = builder.get_graph_info()

        if logger is not None:
            logger.info("[OrtGradientForwardBackward] graph_info=%s",
                        OrtGradientForwardBackward._repr_helper_(
                            graph_info, indent=4))
            logger.info("[OrtGradientForwardBackward] create TrainSession")
            logger.info("[OrtGradientForwardBackward] sess_options=%s",
                        OrtGradientForwardBackward._repr_helper_(
                            self.sess_options, indent=4))
            logger.info(
                "[OrtGradientForwardBackward] providers=%r", self.providers)

        sess = InferenceSession(
            train_onnx_model_serialized, sess_options=self.sess_options,
            provider_options=self.provider_options, providers=self.providers)

        if logger is not None:
            logger.info("[OrtGradientForwardBackward] create InferenceSession")

        sess_eval = InferenceSession(
            optimized_pre_grad_model, sess_options=self.sess_options,
            provider_options=self.provider_options, providers=self.providers)

        if logger is not None:
            logger.info("[OrtGradientForwardBackward] create training agent")

        grad_input_names = [obj.name for obj in sess.get_inputs()]
        bw_fetches_names = [obj.name for obj in sess.get_outputs()]

        fw_outputs_device_info = [
            OrtDevice(
                OrtGradientForwardBackward._provider_name_to_device_type(i),
                OrtDevice.default_memory(), self.device_index)
            for i in self.providers]
        bw_outputs_device_info = [
            OrtDevice(
                OrtGradientForwardBackward._provider_name_to_device_type(
                    self.providers[0]),
                OrtDevice.default_memory(), self.device_index)
            for i in bw_fetches_names]
        fw_no_grad_output_device_info = [
            OrtDevice(
                OrtGradientForwardBackward._provider_name_to_device_type(
                    self.providers[0]),
                OrtDevice.default_memory(), self.device_index)
            for i in self.output_names]

        training_agent = TrainingAgent(
            sess._sess,
            grad_input_names,
            fw_outputs_device_info,
            bw_fetches_names,
            bw_outputs_device_info)

        if logger is not None:
            logger.info(
                "[OrtGradientForwardBackward] instantiate dynamic class %r",
                self.class_name)
            logger.info(
                "[OrtGradientForwardBackward] weights_to_train=%r",
                self.weights_to_train)
            logger.info(
                "[OrtGradientForwardBackward] grad_input_names=%r",
                grad_input_names)
            logger.info(
                "[OrtGradientForwardBackward] bw_fetches_names=%r",
                bw_fetches_names)
            logger.info(
                "[OrtGradientForwardBackward] device_index=%r",
                self.device_index)
        devices = list(fw_outputs_device_info)
        while len(devices) < len(grad_input_names):
            devices.append(devices[-1])

        trained_onnx = onnx.load(BytesIO(train_onnx_model_serialized))
        onnx_loss = onnx.load(BytesIO(optimized_pre_grad_model))
        for i, node in enumerate(trained_onnx.graph.node):
            if node.name == '':
                node.name = "N%d" % i
        for i, node in enumerate(onnx_loss.graph.node):
            if node.name == '':
                node.name = "N%d" % i

        kwargs = {
            '_run_options': self.run_options,
            '_sess': sess,
            '_sess_eval': sess_eval,
            '_training_agent': training_agent,
            '_cache': OrtValueCache(),
            '_logger': logger,
            '_input_names': self.input_names,
            '_grad_input_names': grad_input_names,
            '_output_names': self.output_names,
            '_bw_fetches_names': bw_fetches_names,
            '_fw_outputs_device_info': fw_outputs_device_info,
            '_bw_outputs_device_info': bw_outputs_device_info,
            '_fw_no_grad_output_device_info': fw_no_grad_output_device_info,
            '_weights_to_train': list(sorted(
                self.weights_to_train)),
            '_graph_info': graph_info,
            #
            '_trained_onnx': trained_onnx,
            '_optimized_pre_grad_model': onnx_loss,
            '_graph_builder': builder,
            '_devices': devices,
            '_debug': self.debug
        }
        graph = kwargs['_trained_onnx'].graph
        kwargs.update({
            '_onx_inp': [o.name for o in graph.input],
            '_onx_out': [o.name for o in graph.output]
        })

        if len(kwargs['_onx_inp']) != len(kwargs['_onx_out']):
            raise RuntimeError(  # pragma: no cover
                "Gradient input and output are inconsistant: "
                "%r != %r" % (kwargs['_onx_inp'], kwargs['_onx_out']))
        return kwargs


class OrtGradientForwardBackwardFunction:
    """
    Ancestor for a class implementing forward and backward
    and dynamically created by @see cl OrtGradientForwardBackward.

    Attributes stored in *forward* method:
    * `saved_tensors_`: list of tensors to save during forward
        and to retrieve during backward
    * `state_`: current weights stored in :epkg:`PartialGraphExecutionState`
    """

    def __init__(self):
        self.states_ = []
        self.saved_tensors_ = None

    @staticmethod
    def device_name(device):
        """
        Returns the device name of a device.

        :param device: OrtDevice
        :return: string
        """
        if device.device_type() == OrtDevice.cpu():
            return 'Cpu'
        if device.device_type() == OrtDevice.cuda():
            return 'Gpu'
        raise RuntimeError(  # pragma: no cover
            "Unexpected value for device type %r." % device.device_type())

    @staticmethod
    def input_to_ort(tensors, devices, debug):
        "Converts a list of tensos into an :epkg:`OrtValueVector`."
        def _validate_(tensors):
            if any(map(
                    lambda tu: (
                        tu[0].device_name() !=
                        OrtGradientForwardBackwardFunction.device_name(
                            tu[1])),
                    zip(tensors, devices))):
                raise RuntimeError(  # pragma: no cover
                    "Not all inputs are on the same device %r != %r." % (
                        [OrtGradientForwardBackward.device_name(d)
                         for d in devices],
                        [x.device_name() for x in tensors]))

        if isinstance(tensors, OrtValueVector):
            if debug:
                _validate_(tensors)
            return tensors
        if all(map(lambda t: isinstance(t, C_OrtValue), tensors)):
            if debug:
                _validate_(tensors)
            vect = OrtValueVector()
            vect.reserve(len(tensors))
            for t in tensors:
                if t is None:
                    raise NotImplementedError(  # pragma: no cover
                        "Empty vector found.")
                vect.push_back(t)
            return vect

        # generic case
        vect = OrtValueVector()
        vect.reserve(len(tensors))
        for t, dev in zip(tensors, devices):
            if t is None:
                # if gradient then
                # grad_output = torch.zeros(shape, device=device, dtype=dtype)
                raise NotImplementedError(  # pragma: no cover
                    "Empty vector found.")
            if not t.data.contiguous:
                t = t.as_contiguous()  # pragma: no cover
            vect.push_back(C_OrtValue.ortvalue_from_numpy(t, dev))
        if debug:
            if len(vect) != len(tensors):
                raise RuntimeError(  # pragma: no cover
                    "Unexpected array length %d != %d (len(devices)=%d)." % (
                        len(vect), len(tensors), len(devices)))
            _validate_(vect)
        return vect

    def save_for_backward(self, inputs):
        """
        Saves inputs furing forward steps. The list inputs
        is copied (simple copy, no deep copy).

        :param inputs: list of tensors to save.
        """
        self.saved_tensors_ = list(inputs)

    @property
    def saved_tensors(self):
        """
        Returns saved tensors during forward step.
        """
        if self.saved_tensors_ is None:
            raise RuntimeError(  # pragma: no cover
                "No tensors was saved with save_for_backward.")
        return self.saved_tensors_

    def forward(self, inputs, training=False):
        """
        Implements forward function.

        :param inputs: inputs
        :param training: only inference or training as well
        :return: output as :epkg:`OrtValueVector`
        """
        logger = self._logger
        cls = self.__class__

        def _log(msg, *args):
            logger.debug("[%s.forward] (%dI) " + msg,
                         cls.__name__, len(inputs), *args)

        if logger is not None:
            if training:
                _log("begin with gradient")
            else:
                _log("begin")
            _log("torch function %r", type(cls))
            _log("ort class %r", cls)
            _log("create OrtValueVector (through dlpack)")

        forward_inputs = cls.input_to_ort(
            inputs, cls._devices, cls._debug)

        if training:
            forward_outputs = OrtValueVector()
            state = PartialGraphExecutionState()
            self.states_.append(state)
            if logger is not None:
                _log("run_forward")
            cls._training_agent.run_forward(
                forward_inputs, forward_outputs, state, cls._cache)

            self.save_for_backward(inputs)
            if logger is not None:
                _log("end")
            return forward_outputs
        else:
            # what about bind_input (+ data_ptr)
            if len(forward_inputs) != len(cls._grad_input_names):
                raise RuntimeError(  # pragma: no cover
                    "Size mismatch len(inputs)=%d, len(onnx inputs)=%d." % (
                        len(forward_inputs), len(cls._grad_input_names)))
            iobinding = SessionIOBinding(cls._sess_eval._sess)
            if logger is not None:
                _log("bind inputs %r", cls._grad_input_names)
            for name, inp in zip(
                    cls._grad_input_names, forward_inputs):
                iobinding.bind_ortvalue_input(name, inp)

            # bind output
            if logger is not None:
                _log("bind outputs %r", cls._output_names)
            for name, dev in zip(
                    cls._output_names, cls._fw_no_grad_output_device_info):
                iobinding.bind_output(name, dev)

            # if the shape is known in advance
            # iobinding.bind_output(
            #    output_desc.name, torch_tensor.device.type,
            #    _utils.get_device_index(target_device),
            #    _utils.dtype_torch_to_numpy(torch_tensor.dtype),
            #    list(torch_tensor.size()), torch_tensor.data_ptr())

            if logger is not None:
                _log("grad_enabled=False (run_with_iobinding)")
            cls._sess_eval._sess.run_with_iobinding(
                iobinding, cls._run_options)
            if logger is not None:
                _log("get_outputs")
            ortvalues = iobinding.get_outputs()
            if logger is not None:
                _log("to torck.tensor (%d)", len(ortvalues))
                _log("end")
            return ortvalues

    def backward(self, grad_outputs):
        """
        Implements backward function. The function returns
        an :epkg:`OrtValueVector`.
        """
        cls = self.__class__
        logger = cls._logger

        def _log(msg, *args):
            logger.debug("[%s.backward] (%dI) " + msg,
                         cls.__name__, len(grad_outputs), *args)

        if logger is not None:
            _log("begin")
            _log("torch function %r", type(cls))
            _log("ort class %r", cls)
            _log("saved_tensors")

        inputs = self.saved_tensors
        if logger is not None:
            _log("DEBUG: saved_tensors %r", type(inputs))
            _log("self.state_.pop()")
        state = self.states_.pop()

        if logger is not None:
            _log("create OrtValueVector")

        backward_inputs = cls.input_to_ort(
            grad_outputs, cls._bw_outputs_device_info, cls._debug)

        if logger is not None:
            _log("len(grad_outputs)=%d type(grad_outputs)=%r",
                 len(grad_outputs), type(grad_outputs))
            _log("len(backward_inputs)=%d type(backward_inputs)=%r",
                 len(backward_inputs), type(backward_inputs))
            for i in range(len(backward_inputs)):  # pylint: disable=C0200
                _log("backward_inputs[%d].shape=%r",
                     i, backward_inputs[i].shape())
            _log("run_backward")
        backward_outputs = OrtValueVector()
        cls._training_agent.run_backward(
            backward_inputs, backward_outputs, state)
        if logger is not None:  # pragma: no cover
            _log("DEBUG")
            for i, ov in enumerate(backward_outputs):
                _log("BCK-RET: i=%d - shape=%r - ptr=%r",
                     i, ov.shape(), ov.data_ptr())
            _log("got %r gradients", len(backward_outputs))
            _log("end")
        return backward_outputs
