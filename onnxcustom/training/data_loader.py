"""
@file
@brief Manipulate data for training.
"""
import numpy
from onnxruntime import OrtValue as PyOrtValue


class OrtDataLoader:
    """
    Draws consecutive random observations from a dataset
    by batch. It iterates over the datasets by drawing
    *batch_size* consecutive observations.

    :param X: features
    :param y: labels
    :param batch_size: batch size (consecutive observations)
    :param device: `'cpu'` or `'cuda'`
    :param device_index: device index
    :param random_iter: random iteration

    See example :ref:`l-orttraining-nn-gpu`.
    """

    def __init__(self, X, y, batch_size=20, device='cpu', device_index=0,
                 random_iter=True):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if X.shape[0] != y.shape[0]:
            raise ValueError(  # pragma: no cover
                "Shape mismatch X.shape=%r, y.shape=%r." % (X.shape, y.shape))

        self.X_np = numpy.ascontiguousarray(X)
        self.y_np = numpy.ascontiguousarray(y).reshape((-1, 1))

        self.X_ort = PyOrtValue.ortvalue_from_numpy(
            self.X_np, device, device_index)._ortvalue
        self.y_ort = PyOrtValue.ortvalue_from_numpy(
            self.y_np, device, device_index)._ortvalue

        self.desc = [(self.X_np.shape, self.X_np.dtype),
                     (self.y_np.shape, self.y_np.dtype)]

        self.batch_size = batch_size
        self.device = device
        self.device_index = device_index
        self.random_iter = random_iter

    def __getstate__(self):
        "Removes any non pickable attribute."
        state = {}
        for att in ['X_np', 'y_np', 'desc', 'batch_size',
                    'device', 'device_index', 'random_iter']:
            state[att] = getattr(self, att)
        return state

    def __setstate__(self, state):
        "Restores any non pickable attribute."
        for att, v in state.items():
            setattr(self, att, v)
        self.X_ort = PyOrtValue.ortvalue_from_numpy(
            self.X_np, self.device, self.device_index)._ortvalue
        self.y_ort = PyOrtValue.ortvalue_from_numpy(
            self.y_np, self.device, self.device_index)._ortvalue
        return self

    def __repr__(self):
        "usual"
        return "%s(..., ..., batch_size=%r, device=%r, device_index=%r)" % (
            self.__class__.__name__, self.batch_size, self.device,
            self.device_index)

    def __len__(self):
        "Returns the number of observations."
        return self.desc[0][0][0]

    def _next_iter(self, previous):
        if self.random_iter:
            b = len(self) - self.batch_size
            return numpy.random.randint(0, b)
        if previous == -1:
            return 0
        i = previous + self.batch_size
        if i + self.batch_size > len(self):
            i = len(self) - self.batch_size
        return i

    def iter_numpy(self):
        """
        Iterates over the datasets by drawing
        *batch_size* consecutive observations.
        This iterator is slow as it copies the data of every
        batch. The function yields :eplg:`OrtValue`.
        """
        if self.device not in ('Cpu', 'cpu'):
            raise RuntimeError(
                "Only CPU device is allowed if numpy array are requested "
                "not %r." % self.device)
        N = 0
        b = len(self) - self.batch_size
        if b <= 0 or self.batch_size <= 0:
            yield (self.X_np, self.y_np)
        else:
            i = -1
            while N < len(self):
                i = self._next_iter(i)
                N += self.batch_size
                yield (self.X_np[i:i + self.batch_size],
                       self.y_np[i:i + self.batch_size])

    def iter_ortvalue(self):
        """
        Iterates over the datasets by drawing
        *batch_size* consecutive observations.
        This iterator is slow as it copies the data of every
        batch. The function yields :eplg:`OrtValue`.
        """
        N = 0
        b = len(self) - self.batch_size
        if b <= 0 or self.batch_size <= 0:
            yield (
                PyOrtValue.ortvalue_from_numpy(
                    self.X_np, self.device, self.device_index)._ortvalue,
                PyOrtValue.ortvalue_from_numpy(
                    self.y_np, self.device, self.device_index)._ortvalue)
        else:
            i = -1
            while N < len(self):
                i = self._next_iter(i)
                N += self.batch_size
                xp = self.X_np[i:i + self.batch_size]
                yp = self.y_np[i:i + self.batch_size]
                yield (
                    PyOrtValue.ortvalue_from_numpy(
                        xp, self.device, self.device_index)._ortvalue,
                    PyOrtValue.ortvalue_from_numpy(
                        yp, self.device, self.device_index)._ortvalue)

    def iter_bind(self, bind, names):
        """
        Iterates over the datasets by drawing
        *batch_size* consecutive observations.
        Modifies a bind structure.
        """
        if len(names) != 3:
            raise NotImplementedError(
                "The dataloader expects three (feature name, label name, "
                "learning rate), not %r." % names)

        n_col_x = self.desc[0][0][1]
        n_col_y = self.desc[1][0][1]
        size_x = self.desc[0][1].itemsize
        size_y = self.desc[1][1].itemsize

        def local_bind(bind, offset, n):
            # This function assumes the data is contiguous.
            shape_X = (n, n_col_x)
            shape_y = (n, n_col_y)

            bind.bind_input(
                name=names[0],
                device_type=self.device,
                device_id=self.device_index,
                element_type=self.desc[0][1],
                shape=shape_X,
                buffer_ptr=self.X_ort.data_ptr() + offset * n_col_x * size_x)

            bind.bind_input(
                name=names[1],
                device_type=self.device,
                device_id=self.device_index,
                element_type=self.desc[0][1],
                shape=shape_y,
                buffer_ptr=self.y_ort.data_ptr() + offset * n_col_y * size_y)

        N = 0
        b = len(self) - self.batch_size
        if b <= 0 or self.batch_size <= 0:
            shape_x = self.desc[0][0]
            local_bind(bind, 0, shape_x[0])
            yield shape_x[0]
        else:
            n = self.batch_size
            i = -1
            while N < len(self):
                i = self._next_iter(i)
                N += self.batch_size
                local_bind(bind, i, n)
                yield n

    @property
    def data_np(self):
        "Returns a tuple of the datasets in numpy."
        return self.X_np, self.y_np

    @property
    def data_ort(self):
        "Returns a tuple of the datasets in onnxruntime OrtValue."
        return self.X_ort, self.y_ort
