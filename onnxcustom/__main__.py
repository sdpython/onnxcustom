"""
@file
@brief Implements command line ``python -m onnxcustom <command> <args>``.
"""
import fire
from onnxcustom import check


if __name__ == "__main__":  # pragma: no cover
    fire.Fire({
        'check': check,
    })
