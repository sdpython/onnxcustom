"""
@file
@brief Command lines for profiling.
"""
from ..utils.nvprof2json import convert_trace_to_json


def nvprof2json(filename, output="", temporary_file="", verbose=1,
                fLOG=print):
    """
    Converts traces produced by :epkg:`nvprof` and saved with
    format :epkg:`sqlite3` (extension `.sql`).

    :param filename: filename
    :param output: output file, if left empty, the result is printed
        on the standard output
    :param temporary_file: if the file needs to be unzipped,
        this file will be created to be the unzipped file,
        it is not cleaned after the unzipping.
    :param verbose: verbosity
    :param fLOG: logging function
    :return: json (if output is None, the list of events otherwise)

    .. cmdref::
        :title: Converts a profile stored by nvprof into json
        :cmd: -m onnxcustom nvprof2json --help

        The sqlite dump is generated with a command line similar to:

        ::

            nvprof -o gpu_profile.sql python plot_gpu_training.py

        The command produces a json file following the *Trace Event Format*.
    """
    verbose = int(verbose)
    res = convert_trace_to_json(filename, output, verbose=verbose,
                                temporary_file=temporary_file, fLOG=fLOG)
    if output is None:  # pragma: no cover
        fLOG(res)
