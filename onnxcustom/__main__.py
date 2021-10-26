"""
@file
@brief Implements command line ``python -m onnxcustom <command> <args>``.
"""
import fire
from onnxcustom import check


if __name__ == "__main__":  # pragma: no cover
    from onnxcustom.cli.profiling import nvprof2json
    fire.Fire({
        'check': check,
        'nvprof2json': nvprof2json,
    })
