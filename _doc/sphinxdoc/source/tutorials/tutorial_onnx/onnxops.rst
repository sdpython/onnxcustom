
===========================
ONNX operators and function
===========================

Full list of operators provided by :epkg:`onnx`.
Links point to github page :epkg:`ONNX operators`.

.. runpython::
    :rst:

    import onnx

    fmt = "* `%s <https://github.com/onnx/onnx/blob/main/docs/Operators.md#%s>`_"
    names = list(sorted(set(
        sch.name for sch in onnx.defs.get_all_schemas_with_history())))
    for n in names:
        print(fmt % (n, n))
