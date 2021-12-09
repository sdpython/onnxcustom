"""
@file
@brief Onnxruntime helper.
"""
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtDevice)


def device_to_provider(device_name):
    """
    Converts device into a provider.

    :param device_name: device name (cpu or gpu or cuda)
    :return: provider

    .. runpython::
        :showcode:

        from onnxcustom.utils.onnxruntime_helper import device_to_provider
        print(device_to_provider('cpu'))
    """
    if device_name in ('cpu', 'Cpu'):
        return 'CPUExecutionProvider'
    if device_name in ('Gpu', 'gpu', 'Cuda', 'cuda', 'cuda:0', 'cuda:1'):
        return 'CUDAExecutionProvider'
    raise ValueError(
        "Unexpected value for device_name=%r." % device_name)


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
    Converts device into :epkg:`OrtDevice`.

    :param device: string
    :return: :epkg:`OrtDevice`
    """
    device_type = device if isinstance(device, str) else device.type
    if device_type == 'cuda':
        return OrtDevice.cuda()
    if device_type == 'cpu':
        return OrtDevice.cpu()
    raise ValueError('Unsupported device type: %r.' % device_type)
