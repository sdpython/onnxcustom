A custom converter for a custom model
=====================================

When :epkg:`sklearn-onnx` converts a :epkg:`scikit-learn`
pipeline, it looks into every transformer and predictor
and fetches the associated converter. The resulting
ONNX graph combines the outcome of every converter
in a single graph. If a model does not have its converter,
it displays an error message telling it misses a converter.

.. runpython::
    :showcode:
    

Following section shows how to create your own converter.

.. toctree::
    :maxdepth: 1

    auto_examples/plot_custom_converter
    auto_examples/plot_custom_syntax
    auto_examples/plot_custom_options
    auto_examples/plot_custom_parser
