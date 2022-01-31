
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

    ../../gyexamples/plot_abegin_convert_pipeline
    ../../gyexamples/plot_bbegin_measure_time
    ../../gyexamples/plot_cbegin_opset
    ../../gyexamples/plot_dbegin_options
    ../../gyexamples/plot_dbegin_options_list
    ../../gyexamples/plot_dbegin_options_zipmap
    ../../gyexamples/plot_ebegin_float_double
    ../../gyexamples/plot_funny_sigmoid
    ../../gyexamples/plot_fbegin_investigate
    ../../gyexamples/plot_gbegin_dataframe
    ../../gyexamples/plot_gbegin_transfer_learning
    ../../gyexamples/plot_gbegin_cst
    ../../gyexamples/plot_gconverting
