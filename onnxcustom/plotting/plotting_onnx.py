# flake8: noqa: F401
"""
@file
@brief Shortcuts to plotting.
"""
from mlprodict.plotting.plotting_onnx import plot_onnx


def plot_onnxs(*onx, ax=None, dpi=300, temp_dot=None, temp_img=None,
               show=False, title=None):
    """
    Plots one or several ONNX graph into a :epkg:`matplotlib` graph.

    :param onx: ONNX objects
    :param ax: existing axes
    :param dpi: resolution
    :param temp_dot: temporary file,
        if None, a file is created and removed
    :param temp_img: temporary image,
        if None, a file is created and removed
    :param show: calls `plt.show()`
    :param title: graph title
    :return: axes
    """
    if len(onx) == 1:
        if ax is None:
            import matplotlib.pyplot as plt  # pylint: disable=C0415
            ax = plt.gca()
        elif isinstance(ax, str) and ax == 'new':
            import matplotlib.pyplot as plt  # pylint: disable=C0415
            _, ax = plt.subplots(1, 1)
        ax = plot_onnx(onx[0], ax=ax, dpi=dpi, temp_dot=temp_dot,
                       temp_img=temp_img)
        if title is not None:
            ax.set_title(title)
        return ax
    elif len(onx) > 1 and isinstance(ax, str) and ax == 'new':
        ax = None

    if len(onx) == 0:
        raise ValueError(
            "Empty list of graph to plot.")

    if ax is None:
        import matplotlib.pyplot as plt  # pylint: disable=C0415
        fig, ax = plt.subplots(1, len(onx))
    else:
        fig = None
    if ax.shape[0] != len(onx):
        raise ValueError(
            "ax must be an array of shape (%d, )." % len(onx))
    for i, ox in enumerate(onx):
        plot_onnx(ox, ax=ax[i], dpi=dpi, temp_dot=temp_dot,
                  temp_img=temp_img)
        if title is None or isinstance(title, str):
            continue
        if i < len(title):
            ax[i].set_title(title[i])
    if len(onx) > 1 and isinstance(title, str):
        if fig is None:
            raise ValueError(  # pragma: no cover
                "Main title cannot be set if fig is undefined (title=%r, "
                "len(onx)=%d)" % (title, len(onx)))
        fig.suptitle(title)
    elif len(onx) == 1:
        ax.set_title(title)
    return ax
