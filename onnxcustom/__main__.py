# pylint: disable=C0415
"""
@file
@brief Implements command line ``python -m onnxcustom <command> <args>``.
"""


def main(args, fLOG=print):
    """
    Implements ``python -m onnxcustom <command> <args>``.

    :param args: command line arguments
    :param fLOG: logging function
    """
    from pyquickhelper.cli import cli_main_helper
    try:
        from . import check
        from .cli.profiling import nvprof2json
    except ImportError:  # pragma: no cover
        from onnxcustom import check
        from onnxcustom.cli.profiling import nvprof2json

    fcts = dict(nvprof2json=nvprof2json, check=check)
    return cli_main_helper(fcts, args=args, fLOG=fLOG)


if __name__ == "__main__":
    import sys  # pragma: no cover
    main(sys.argv[1:])  # pragma: no cover
