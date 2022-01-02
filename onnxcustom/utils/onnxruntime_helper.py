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
        "Unexpected value for provider_name=%r." % provider_name)


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
            'Unsupported device type: %r.' % device)
    if not hasattr(device, 'type'):
        raise TypeError('Unsupported device type: %r.' % type(device))
    device_type = device.type
    if device_type == 'cuda':
        return C_OrtDevice.cuda()
    if device_type == 'cpu':
        return C_OrtDevice.cpu()
    raise ValueError(  # pragma: no cover
        'Unsupported device type: %r.' % device_type)


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
            "Unable to interpret string %r as a device." % device)
    raise TypeError(  # pragma: no cover
        "Unable to interpret type %r, (%r) as de device." % (
            type(device), device))


def ort_device_to_string(device):
    """
    Returns a string representing the device.
    Opposite of function @see fn get_ort_device.

    :param device: see :epkg:`C_OrtDevice`
    :return: string
    """
    if not isinstance(device, C_OrtDevice):
        raise TypeError(
            "device must be of type C_OrtDevice not %r." % type(device))
    ty = device.device_type()
    if ty == C_OrtDevice.cpu():
        sty = 'cpu'
    elif ty == C_OrtDevice.cuda():
        sty = 'cuda'
    else:
        raise NotImplementedError(  # pragma: no cover
            "Unable to guess device for %r and type=%r." % (device, ty))
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
        "Unexpected device %r." % device)
