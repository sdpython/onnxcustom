"""
@file
@brief Manipulate data for training.
"""
import numpy
from onnxruntime import OrtValue


class OrtDataLoader:
    """
    Draws consecutive random observations from a dataset
    by batch. It iterates over the datasets by drawing
    *batch_size* consecutive observations.

    :param X: features
    :param y: labels
    :param batch_size: batch size (consecutive observations)
    :param device: `'cpu'` or `'cuda'`
    :param device_idx: device index

    See example :ref:`l-orttraining-nn-gpu`.
    """

    def __init__(self, X, y, batch_size=20, device='cpu', device_idx=0):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "Shape mismatch X.shape=%r, y.shape=%r." % (X.shape, y.shape))
        self.X = numpy.ascontiguousarray(X)
        self.y = numpy.ascontiguousarray(y)
        self.batch_size = batch_size
        self.device = device
        self.device_idx = device_idx

    def __repr__(self):
        "usual"
        return "%s(..., ..., batch_size=%r, device=%r, device_idx=%r)" % (
            self.__class__.__name__, self.batch_size, self.device,
            self.device_idx)

    def __len__(self):
        "Returns the number of observations."
        return self.X.shape[0]

    def __iter__(self):
        """
        Iterates over the datasets by drawing
        *batch_size* consecutive observations.
        """
        N = 0
        b = len(self) - self.batch_size
        if b <= 0 or self.batch_size <= 0:
            yield (
                OrtValue.ortvalue_from_numpy(
                    self.X, self.device, self.device_idx),
                OrtValue.ortvalue_from_numpy(
                    self.y, self.device, self.device_idx))
        else:
            while N < len(self):
                i = numpy.random.randint(0, b)
                N += self.batch_size
                yield (
                    OrtValue.ortvalue_from_numpy(
                        self.X[i:i + self.batch_size],
                        self.device, self.device_idx),
                    OrtValue.ortvalue_from_numpy(
                        self.y[i:i + self.batch_size],
                        self.device, self.device_idx))

    @property
    def data(self):
        "Returns a tuple of the datasets."
        return self.X, self.y
