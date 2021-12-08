
mappings
========

.. contents::
    :local:

Type Mappings
+++++++++++++

.. _l-onnx-types-mapping:

TENSOR_TYPE_TO_NP_TYPE
~~~~~~~~~~~~~~~~~~~~~~

.. runpython::
    :showcode:

    import pprint
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

    pprint.pprint(TENSOR_TYPE_TO_NP_TYPE)

NP_TYPE_TO_TENSOR_TYPE
~~~~~~~~~~~~~~~~~~~~~~

.. runpython::
    :showcode:

    import pprint
    from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

    pprint.pprint(NP_TYPE_TO_TENSOR_TYPE)

TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. runpython::
    :showcode:

    import pprint
    from onnx.mapping import TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE

    pprint.pprint(TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE)

STORAGE_TENSOR_TYPE_TO_FIELD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. runpython::
    :showcode:

    import pprint
    from onnx.mapping import STORAGE_TENSOR_TYPE_TO_FIELD

    pprint.pprint(STORAGE_TENSOR_TYPE_TO_FIELD)

STORAGE_ELEMENT_TYPE_TO_FIELD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. runpython::
    :showcode:

    import pprint
    from onnx.mapping import STORAGE_ELEMENT_TYPE_TO_FIELD

    pprint.pprint(STORAGE_ELEMENT_TYPE_TO_FIELD)

OPTIONAL_ELEMENT_TYPE_TO_FIELD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. runpython::
    :showcode:

    import pprint
    from onnx.mapping import OPTIONAL_ELEMENT_TYPE_TO_FIELD

    pprint.pprint(OPTIONAL_ELEMENT_TYPE_TO_FIELD)

Opset Version
+++++++++++++

.. autofunction:: onnx.defs.onnx_opset_version

.. autofunction:: onnx.defs.get_all_schemas_with_history

Internal module
+++++++++++++++

.. automodule:: onnx.onnx_cpp2py_export.defs
    :members:
