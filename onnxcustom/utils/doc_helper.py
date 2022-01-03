"""
@file
@brief Helpers to improve documentation rendering.
"""
import re


def fix_link_operator_md(markdown):
    """
    The redering of file `Operator.md <https://github.com/onnx/onnx/
    blob/master/docs/Operators.md>`_ breaks links. This function
    restores some of them.

    :param markdown: a string or a filename
    :return: modified content
    """
    if len(markdown) < 5000 and markdown.endswith('.md'):
        with open(markdown, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = markdown  # pragma: no cover

    reg = re.compile(
        "([|]<a href=\\\"#(?P<name>[.A-Za-z]+)\\\">(?P=name)</a>[|])")
    pattern = "|[{1}](#a-name-{0}-a-a-name-{0}-{0}-a)|"

    lines = content.split('\n')
    new_lines = []
    for line in lines:
        find = reg.search(line)
        if find:
            gr = find.groups()
            exp = gr[0]
            op = gr[1]
            rep = pattern.format(op.lower(), op).replace(".", "-")
            line = line.replace(exp, rep)
        new_lines.append(line)
    return "\n".join(new_lines)
