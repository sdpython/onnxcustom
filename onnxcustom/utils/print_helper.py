"""
@file
@brief Helpers to display internal structures.
"""


def str_ortvalue(ov):
    """
    Displays the content of an :epkg:`C_OrtValue`.

    :param ov: :epkg:`OrtValue` or :epkg:`C_OrtValue`
    :return: str
    """
    if hasattr(ov, '_ortvalue'):
        return str_ortvalue(ov._ortvalue)
    device = ov.device_name()
    value = ov.numpy()
    values = value.ravel().tolist()
    if len(values) > 10:
        values = values[:5] + ["..."] + values[-5:]
    return "device=%s dtype=%r shape=%r value=%r" % (
        device, value.dtype, value.shape, values)
