"""
@file
@brief Manipulate data for training.
"""
import numpy
from ..utils.onnxruntime_helper import (
    get_ort_device, numpy_to_ort_value, ort_device_to_string)


class OrtDataLoader:
    """
    Draws consecutive random observations from a dataset
    by batch. It iterates over the datasets by drawing
    *batch_size* consecutive observations.

    :param X: features
    :param y: labels
    :param sample_weight: weight or None
    :param batch_size: batch size (consecutive observations)
    :param device: :epkg:`C_OrtDevice` or a string such as `'cpu'`
    :param random_iter: random iteration

    See example :ref:`l-orttraining-nn-gpu`.
    """

    def __init__(self, X, y, sample_weight=None,
                 batch_size=20, device='cpu', random_iter=True):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if X.shape[0] != y.shape[0]:
            raise ValueError(  # pragma: no cover
                "Shape mismatch X.shape=%r, y.shape=%r." % (X.shape, y.shape))

        self.batch_size = batch_size
        self.device = get_ort_device(device)
        self.random_iter = random_iter

        self.X_np = numpy.ascontiguousarray(X)
        self.y_np = numpy.ascontiguousarray(y).reshape((-1, 1))

        self.X_ort = numpy_to_ort_value(self.X_np, self.device)
        self.y_ort = numpy_to_ort_value(self.y_np, self.device)

        self.desc = [(self.X_np.shape, self.X_np.dtype),
                     (self.y_np.shape, self.y_np.dtype)]

        if sample_weight is None:
            self.w_np = None
            self.w_ort = None
        else:
            if X.shape[0] != sample_weight.shape[0]:
                raise ValueError(  # pragma: no cover
                    "Shape mismatch X.shape=%r, sample_weight.shape=%r."
                    "" % (X.shape, sample_weight.shape))
            self.w_np = numpy.ascontiguousarray(
                sample_weight).reshape((-1, ))
            self.w_ort = numpy_to_ort_value(self.w_np, self.device)
            self.desc.append((self.w_np.shape, self.w_np.dtype))

    def __getstate__(self):
        "Removes any non pickable attribute."
        state = {}
        for att in ['X_np', 'y_np', 'w_np',
                    'desc', 'batch_size', 'random_iter']:
            state[att] = getattr(self, att)
        state['device'] = ort_device_to_string(self.device)
        return state

    def __setstate__(self, state):
        "Restores any non pickable attribute."
        for att, v in state.items():
            setattr(self, att, v)
        self.device = get_ort_device(self.device)
        self.X_ort = numpy_to_ort_value(self.X_np, self.device)
        self.y_ort = numpy_to_ort_value(self.y_np, self.device)
        if self.w_np is None:
            self.w_ort = None
        else:
            self.w_ort = numpy_to_ort_value(
                self.w_np, self.device)
        return self

    def __repr__(self):
        "usual"
        return "%s(..., ..., batch_size=%r, device=%r)" % (
            self.__class__.__name__, self.batch_size,
            ort_device_to_string(self.device))

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
        batch. The function yields :epkg:`C_OrtValue`.
        """
        if self.device.device_type() != self.device.cpu():
            raise RuntimeError(  # pragma: no cover
                "Only CPU device is allowed if numpy arrays are requested "
                "not %r." % ort_device_to_string(self.device))
        N = 0
        b = len(self) - self.batch_size
        if self.w_np is None:
            if b <= 0 or self.batch_size <= 0:
                yield (self.X_np, self.y_np)
            else:
                i = -1
                while N < len(self):
                    i = self._next_iter(i)
                    N += self.batch_size
                    yield (self.X_np[i:i + self.batch_size],
                           self.y_np[i:i + self.batch_size])
        else:
            if b <= 0 or self.batch_size <= 0:
                yield (self.X_np, self.y_np, self.w_np)
            else:
                i = -1
                while N < len(self):
                    i = self._next_iter(i)
                    N += self.batch_size
                    yield (self.X_np[i:i + self.batch_size],
                           self.y_np[i:i + self.batch_size],
                           self.w_np[i:i + self.batch_size])

    def iter_ortvalue(self):
        """
        Iterates over the datasets by drawing
        *batch_size* consecutive observations.
        This iterator is slow as it copies the data of every
        batch. The function yields :epkg:`C_OrtValue`.
        """
        N = 0
        b = len(self) - self.batch_size
        if self.w_ort is None:
            if b <= 0 or self.batch_size <= 0:
                yield (self.X_ort, self.y_ort)
            else:
                i = -1
                while N < len(self):
                    i = self._next_iter(i)
                    N += self.batch_size
                    xp = self.X_np[i:i + self.batch_size]
                    yp = self.y_np[i:i + self.batch_size]
                    yield (
                        numpy_to_ort_value(xp, self.device),
                        numpy_to_ort_value(yp, self.device))
        else:
            if b <= 0 or self.batch_size <= 0:
                yield (self.X_ort, self.y_ort, self.w_ort)
            else:
                i = -1
                while N < len(self):
                    i = self._next_iter(i)
                    N += self.batch_size
                    xp = self.X_np[i:i + self.batch_size]
                    yp = self.y_np[i:i + self.batch_size]
                    wp = self.w_np[i:i + self.batch_size]
                    yield (
                        numpy_to_ort_value(xp, self.device),
                        numpy_to_ort_value(yp, self.device),
                        numpy_to_ort_value(wp, self.device))

    def iter_bind(self, bind, names):
        """
        Iterates over the datasets by drawing
        *batch_size* consecutive observations.
        Modifies a bind structure.
        """
        if len(names) not in (3, 4):
            raise NotImplementedError(
                "The dataloader expects three (feature name, label name, "
                "learning rate) or (feature name, label name, sample_weight, "
                "learning rate), not %r." % names)

        n_col_x = self.desc[0][0][1]
        n_col_y = self.desc[1][0][1]
        size_x = self.desc[0][1].itemsize
        size_y = self.desc[1][1].itemsize
        size_w = None if len(self.desc) <= 2 else self.desc[2][1].itemsize

        def local_bind(bind, offset, n):
            # This function assumes the data is contiguous.
            shape_X = (n, n_col_x)
            shape_y = (n, n_col_y)

            bind.bind_input(
                names[0], self.device, self.desc[0][1], shape_X,
                self.X_ort.data_ptr() + offset * n_col_x * size_x)
            bind.bind_input(
                names[1], self.device, self.desc[1][1], shape_y,
                self.y_ort.data_ptr() + offset * n_col_y * size_y)

        def local_bindw(bind, offset, n):
            # This function assumes the data is contiguous.
            shape_w = (n, )

            bind.bind_input(
                names[2], self.device, self.desc[2][1], shape_w,
                self.w_ort.data_ptr() + offset * size_w)

        N = 0
        b = len(self) - self.batch_size
        if self.w_ort is None:
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
        else:
            if b <= 0 or self.batch_size <= 0:
                shape_x = self.desc[0][0]
                local_bind(bind, 0, shape_x[0])
                local_bindw(bind, 0, shape_x[0])
                yield shape_x[0]
            else:
                n = self.batch_size
                i = -1
                while N < len(self):
                    i = self._next_iter(i)
                    N += self.batch_size
                    local_bind(bind, i, n)
                    local_bindw(bind, i, n)
                    yield n

    @property
    def data_np(self):
        "Returns a tuple of the datasets in numpy."
        if self.w_np is None:
            return self.X_np, self.y_np
        return self.X_np, self.y_np, self.w_np

    @property
    def data_ort(self):
        "Returns a tuple of the datasets in onnxruntime C_OrtValue."
        if self.w_ort is None:
            return self.X_ort, self.y_ort
        return self.X_ort, self.y_ort, self.w_ort
