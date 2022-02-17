
========
OrtValue
========

:epkg:`numpy` has its :class:`numpy.ndarray`, :epkg:`pytorch` has
its :class:`torch.Tensor`. :epkg:`onnxruntime` has its
`OrtValue`. As opposed to the other two framework,
`OrtValue` does not support simple operations such as
addition, subtraction, multiplication or division. It can only be
used to be consumed by :epkg:`onnxruntime` or converted into another
object such as :class:`numpy.ndarray`. An `OrtValue` can hold more than
a dense tensor, it can also be a sparse tensor, a sequence of tensors
or a map of tensors. Like :class:`torch.Tensor`, the data can be located
on CPU, CUDA, ...

.. contents::
    :local:

.. note::
    :epkg:`onnxruntime` implements a C class named `OrtValue`
    but referred as :epkg:`C_OrtValue`
    and a python wrapper for it also named :epkg:`OrtValue`.
    This documentation uses :epkg:`C_OrtValue` directly.
    The wrapper is usually calling the same C functions.
    The same goes for :epkg:`OrtDevice` and :epkg:`C_OrtDevice`.
    They can be imported like this:

    ::

        from onnxruntime.capi._pybind_state import (
            OrtValue as C_OrtValue,
            OrtDevice as C_OrtDevice)

.. _l-doc-device:

Device
======

A device is associated to a tensor. It indicates
where the data is stored. It is defined by:

* a device type: CPU, CUDA, FGPA
* a device index: if there are many devices of the
  same type, it tells which one is used.
* an allocator: it is possible to change the way
  memory is allocated.

Next example shows how to create a CPU device.

.. runpython::
    :showcode:

    from onnxruntime.capi._pybind_state import (
        OrtDevice as C_OrtDevice)

    ort_device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)

    print(ort_device)
    print(ort_device.device_type(), C_OrtDevice.cpu())

And the next one how to create a CUDA device.

.. runpython::
    :showcode:

    from onnxruntime.capi._pybind_state import (
        OrtDevice as C_OrtDevice)

    ort_device = C_OrtDevice(
        C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)

    print(ort_device)
    print(ort_device.device_type(), C_OrtDevice.cuda())

The class has three methods:

* *device_type()*: returns the device type
* *device_id()*: returns the device index
* *device_mem_type()*: *not available yet*

Memory Allocator
================

.. index:: incomplete

**to be continued later**

OrtValue
========

This class is a generic type. It hides any supported type
by :epkg:`onnxruntime`, a tensor, a sparse tensor,
a sequence of tensors, a map of tensors. From python point of view,
it is only a container. It is only possible to export,
convert or get information about it. The only way to manipulate
*OrtValue* is to go through an ONNX graph loaded by
an :epkg:`InferenceSession`.
Following section refers to the C implementation of :epkg:`C_OrtValue`.

Creation from numpy
+++++++++++++++++++

The most easier way is to create an :epkg:`C_OrtValue` from
a :class:`numpy.ndarray`. Next example does that on CPU.
However even that simple example hides some important detail.

.. runpython::
    :showcode:

    import numpy
    from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
        OrtValue as C_OrtValue,
        OrtDevice as C_OrtDevice,
        OrtMemType)
    from onnxcustom.utils.print_helper import str_ortvalue

    vect = numpy.array([100, 100], dtype=numpy.float32)

    device = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
    ort_value = C_OrtValue.ortvalue_from_numpy(vect, device)
    print(ort_value)
    print(str_ortvalue(ort_value))

    # Data pointers?
    print(ort_value.data_ptr())
    print(vect.__array_interface__['data'])

The last two lines show that both objects point to the same location.
To avoid copying the data, :epkg:`onnxruntime` only creates a structure
wrapping the same memory buffer. As a result, the numpy array must
**remain alive** as long as the instance of `C_OrtValue` is.
If it does not, the program usually crashes with no exception but a
segmentation fault.

Creation from a new buffer
++++++++++++++++++++++++++

Method `ortvalue_from_shape_and_type` can create a new
:epkg:`C_OrtValue` owning its buffer.

.. runpython::
    :showcode:

    import numpy
    from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
        OrtValue as C_OrtValue,
        OrtDevice as C_OrtDevice,
        OrtMemType)
    from onnxcustom.utils.print_helper import str_ortvalue

    device = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
    ort_value = C_OrtValue.ortvalue_from_shape_and_type(
        [100, 100], numpy.float32, device)

    print(ort_value)
    print(str_ortvalue(ort_value))

    # Address can be given to another C function to populate the buffer.
    print(ort_value.data_ptr())

Export to numpy
+++++++++++++++

Unless it is reused by another library or :epkg:`onnxruntime`
itself, the only way to access the data it contains is to
create a numpy array with method `numpy`.

.. runpython::
    :showcode:

    import numpy
    from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
        OrtValue as C_OrtValue,
        OrtDevice as C_OrtDevice,
        OrtMemType)
    from onnxcustom.utils.print_helper import str_ortvalue

    vect = numpy.array([100, 100], dtype=numpy.float32)

    device = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
    ort_value = C_OrtValue.ortvalue_from_numpy(vect, device)
    print(ort_value)
    print(str_ortvalue(ort_value))

    # Data pointers?
    print(ort_value.data_ptr())
    print(vect.__array_interface__['data'])

    # to numpy
    vect2 = ort_value.numpy()
    print(vect2.__array_interface__['data'])

Method `numpy` makes a copy. Next section brings more details
about avoiding that copy.

DLPack
======

:epkg:`DLPack` is a protocol imagined to avoid copying memory when data
is created by one framework and used by another one. The safest way is
to copy entirely the data in its own containers. But that costs a lot
if the data is big or may be even difficult if the data is big compared
to the memory size. The DLpack structure describes a tensor, or a multidimensional
vector with a specific element type and a specific shape. It also
keeps the location or device where the data is (CPU, CUDA, ...).
When a library B receives a DLpack structure from a library A, it:

* creates its own to store any information it needs
* it deletes the structure it receives by calling a destructor
  store in the structure itself.

The library B takes ownership of the data and is now responsible for
its deletion unless a library C requests its ownership through a DLpack
structure as well.

:epkg:`pytorch` implements this through two functions `to_dlpack` and
`from_dlpack` (see `torch.utils.dlpack
<https://pytorch.org/docs/stable/dlpack.html>`_).
:epkg:`numpy` implements it as well. The changes were merged in
`PR 19083 <https://github.com/numpy/numpy/pull/19083>`_.

:epkg:`onnxruntime-training` implements a couple of scenarios based
on :epkg:`pytorch` and needs this protocol to avoid unnecessary
data transfer.

Conversion
++++++++++

Method `to_dlpack` exports a :epkg:`C_OrtValue` into a DLPack stucture.
Static method `from_dlpack` creates :epkg:`C_OrtValue` from a DLPack stucture.
Everytime one of these methods is used, the previous container loses
ownership to the next one. Only this one must be used. It becomes
responsible for the data deletion.

.. runpython::
    :showcode:

    import numpy
    from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
        OrtValue as C_OrtValue,
        OrtDevice as C_OrtDevice,
        OrtMemType)
    from onnxcustom.utils.print_helper import str_ortvalue

    vect = numpy.array([100, 100], dtype=numpy.float32)
    device = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
    ort_value = C_OrtValue.ortvalue_from_numpy(vect, device)
    print("ptr", ort_value.data_ptr())

    # export
    dlp = ort_value.to_dlpack()
    print(dlp)

    # export back to onnxruntime
    ort_value_back = C_OrtValue.from_dlpack(dlp, False)
    # dlp structure is no longer valid
    print("ptr", ort_value_back.data_ptr())
    print(str_ortvalue(ort_value_back))

.. index:: incomplete

**to be continued later**

See `PR 9610
<https://github.com/microsoft/onnxruntime/pull/9610>`_.

OrtValueVector
++++++++++++++

This container is equivalent to a list of :epkg:`C_OrtValue`.
It optimizes the conversion to DLPack structure (see `PR 9610
<https://github.com/microsoft/onnxruntime/pull/9610>`_).

.. index:: incomplete

**to be continued later**

Boolean ambiguity
+++++++++++++++++

Boolean type is usually represented as a vector of unsigned bytes.
This information is not actually stored in the DLPack structure
and there is no way to distinguish between the two. That's why
method `from_dlpack` has an additional parameter. You can read
more about this in `issue 75 <https://github.com/dmlc/dlpack/issues/75>`_.

Sparse Tensors
==============

Sparse tensors only represent 2D matrices and are much more efficient
in standard machine learning to represent categories or text features.
This structure is usually created by an operator such as
:epkg:`OneHotEncoder` or :epkg:`TfIdfVectorizer`.

CSR
+++

The following example shows how to create a sparse tensor
(C version, :epkg:`C_SparseTensor`) from a :epkg:`CSR` matrix
and to convert it back to this format.

.. runpython::
    :showcode:

    import numpy
    from scipy.sparse import csr_matrix
    from onnxruntime.capi._pybind_state import (
        SparseTensor as C_SparseTensor,
        OrtDevice as C_OrtDevice)

    ort_device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)

    dense = (numpy.random.randn(100, 10) >= 2).astype(numpy.float32)
    print("sparse ratio:", dense.sum() * 1.0 / dense.size)

    csr = csr_matrix(dense)
    print("csr_matrix:")
    print(csr)

    ort_sparse = C_SparseTensor.sparse_csr_from_numpy(
        csr.shape,
        csr.data, csr.indices, csr.indptr,
        ort_device)

    print("ort_sparse.values() ->", ort_sparse.values())

    # Back to csr_matrix.
    ort_csr = ort_sparse.get_csrc_data()

    csr2 = csr_matrix(
        (ort_sparse.values(), ort_csr.inner(), ort_csr.outer()),
        shape=ort_sparse.dense_shape())

    print("retrieved:")
    print(csr2)

COO
+++

Previous example was changed to do the same with format
:epkg:`COO`.

.. runpython::
    :showcode:

    import numpy
    from scipy.sparse import coo_matrix
    from onnxruntime.capi._pybind_state import (
        SparseTensor as C_SparseTensor,
        OrtDevice as C_OrtDevice)

    ort_device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)

    dense = (numpy.random.randn(100, 10) >= 2).astype(numpy.float32)
    print("sparse ratio:", dense.sum() * 1.0 / dense.size)

    coo = coo_matrix(dense)
    print("coo_matrix:")
    print(coo)

    ort_sparse = C_SparseTensor.sparse_coo_from_numpy(
        coo.shape,
        coo.data,
        numpy.hstack([coo.row.reshape((-1, 1)), coo.col.reshape((-1, 1))]),
        ort_device)

    print("ort_sparse.values() ->", ort_sparse.values())

    # Back to coo_matrix.
    ort_coo = ort_sparse.get_coo_data()

    indices = ort_coo.indices()
    coo2 = coo_matrix(
        (ort_sparse.values(), (indices[:, 0], indices[:, 1])),
        shape=ort_sparse.dense_shape())

    print("retrieved:")
    print(coo2)
