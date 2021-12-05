"""
@file
@brief Helpers to improve documentation rendering.
"""


def fix_link_operator_md(markdown):
    """
    The redering of file `Operator.md <https://github.com/onnx/onnx/
    blob/master/docs/Operators.md>`_ breaks links. This function
    restores some of them.

    :param markdown: a string or a filename
    :return: modified content
    """
    if len(markdown) < 5000 and markdown.endwith('.md'):
        with open(markdown, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = markdown
    return content
