
The easy case
=============

The easy case is when the machine learned model
can be converter into ONNX with a converting library
without writing nay specific code. That means that a converter
exists for the model or each piece of the model,
the converter produces an ONNX graph where every node
is part of the existing ONNX specifications, the runtime
used to compute the predictions implements every node
used in the ONNX graph.

.. toctree::
    :maxdepth: 1

    auto_examples/plot_begin_convert_pipeline
    auto_examples/plot_begin_measure_time
    auto_examples/plot_begin_opset
    auto_examples/plot_begin_options
    auto_examples/plot_begin_float_double
    auto_examples/plot_begin_investigate
