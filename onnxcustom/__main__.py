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
    from pyquickhelper.cli import cli_main_helper  # pylint: disable=C0415
    try:
        from . import check  # pylint: disable=C0415
        from .cli.profiling import nvprof2json  # pylint: disable=C0415
    except ImportError:  # pragma: no cover
        from onnxcustom import check  # pylint: disable=C0415
        from onnxcustom.cli.profiling import nvprof2json  # pylint: disable=C0415

    fcts = dict(nvprof2json=nvprof2json, check=check)
    return cli_main_helper(fcts, args=args, fLOG=fLOG)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])  # pragma: no cover
