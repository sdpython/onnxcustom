"""
@file
@brief Onnxruntime helper.
"""
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtDevice as C_OrtDevice, OrtValue as C_OrtValue)


def provider_to_device(provider_name):
    """
    Converts provider into a device.

    :param provider_name: provider name
    :return: device name

    .. runpython::
        :showcode:

        from onnxcustom.utils.onnxruntime_helper import provider_to_device
        print(provider_to_device('CPUExecutionProvider'))
    """
    if provider_name == 'CPUExecutionProvider':
        return 'cpu'
    if provider_name == 'CUDAExecutionProvider':
        return 'cuda'
    raise ValueError(
        f"Unexpected value for provider_name={provider_name!r}.")


def get_ort_device_type(device):
    """
    Converts device into device type.

    :param device: string
    :return: device type
    """
    if isinstance(device, str):
        if device == 'cuda':
            return C_OrtDevice.cuda()
        if device == 'cpu':
            return C_OrtDevice.cpu()
        raise ValueError(  # pragma: no cover
            f'Unsupported device type: {device!r}.')
    if not hasattr(device, 'device_type'):
        raise TypeError(f'Unsupported device type: {type(device)!r}.')
    device_type = device.device_type()
    if device_type in ('cuda', 1):
        return C_OrtDevice.cuda()
    if device_type in ('cpu', 0):
        return C_OrtDevice.cpu()
    raise ValueError(  # pragma: no cover
        f'Unsupported device type: {device_type!r}.')


def get_ort_device(device):
    """
    Converts device into :epkg:`C_OrtDevice`.

    :param device: any type
    :return: :epkg:`C_OrtDevice`

    Example:

    ::

        get_ort_device('cpu')
        get_ort_device('gpu')
        get_ort_device('cuda')
        get_ort_device('cuda:0')
    """
    if isinstance(device, C_OrtDevice):
        return device
    if isinstance(device, str):
        if device == 'cpu':
            return C_OrtDevice(
                C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
        if device in {'gpu', 'cuda:0', 'cuda', 'gpu:0'}:
            return C_OrtDevice(
                C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)
        if device.startswith('gpu:'):
            idx = int(device[4:])
            return C_OrtDevice(
                C_OrtDevice.cuda(), C_OrtDevice.default_memory(), idx)
        if device.startswith('cuda:'):
            idx = int(device[5:])
            return C_OrtDevice(
                C_OrtDevice.cuda(), C_OrtDevice.default_memory(), idx)
        raise ValueError(
            f"Unable to interpret string {device!r} as a device.")
    raise TypeError(  # pragma: no cover
        f"Unable to interpret type {type(device)!r}, ({device!r}) as de device.")


def ort_device_to_string(device):
    """
    Returns a string representing the device.
    Opposite of function @see fn get_ort_device.

    :param device: see :epkg:`C_OrtDevice`
    :return: string
    """
    if not isinstance(device, C_OrtDevice):
        raise TypeError(
            f"device must be of type C_OrtDevice not {type(device)!r}.")
    ty = device.device_type()
    if ty == C_OrtDevice.cpu():
        sty = 'cpu'
    elif ty == C_OrtDevice.cuda():
        sty = 'cuda'
    else:
        raise NotImplementedError(  # pragma: no cover
            f"Unable to guess device for {device!r} and type={ty!r}.")
    idx = device.device_id()
    if idx == 0:
        return sty
    return "%s:%d" % (sty, idx)


def numpy_to_ort_value(arr, device=None):
    """
    Converts a numpy array to :epkg:`C_OrtValue`.

    :param arr: numpy array
    :param device: :epkg:`C_OrtDevice` or None for cpu
    :return: :epkg:`C_OrtValue`
    """
    if device is None:
        device = get_ort_device('cpu')
    return C_OrtValue.ortvalue_from_numpy(arr, device)


def device_to_providers(device):
    """
    Returns the corresponding providers for a specific device.

    :param device: :epkg:`C_OrtDevice`
    :return: providers
    """
    if isinstance(device, str):
        device = get_ort_device(device)
    if device.device_type() == device.cpu():
        return ['CPUExecutionProvider']
    if device.device_type() == device.cuda():
        return ['CUDAExecutionProvider']
    raise ValueError(  # pragma: no cover
        f"Unexpected device {device!r}.")


def get_ort_device_from_session(sess):
    """
    Retrieves the device from an object :epkg:`InferenceSession`.

    :param sess: :epkg:`InferenceSession`
    :return: :epkg:`C_OrtDevice`
    """
    providers = sess.get_providers()
    if providers == ["CPUExecutionProvider"]:
        return C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    if providers[0] == "CUDAExecutionProvider":
        options = sess.get_provider_options()
        if len(options) == 0:
            return C_OrtDevice(C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)
        if "CUDAExecutionProvider" not in options:
            raise NotImplementedError(
                f"Unable to guess 'device_id' in {options}.")
        cuda = options["CUDAExecutionProvider"]
        if "device_id" not in cuda:
            raise NotImplementedError(
                f"Unable to guess 'device_id' in {options}.")
        device_id = int(cuda["device_id"])
        return C_OrtDevice(C_OrtDevice.cuda(), C_OrtDevice.default_memory(), device_id)
    raise NotImplementedError(
        f"Not able to guess the model device from {providers}.")
