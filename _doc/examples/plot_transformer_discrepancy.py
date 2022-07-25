"""
.. _example-transform-discrepancy:

Dealing with discrepancies (tf-idf)
===================================

.. index:: td-idf

`TfidfVectorizer <https://scikit-learn.org/stable/modules/
generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_
is one transform for which the corresponding converted onnx model
may produce different results. The larger the vocabulary is,
the higher the probability to get different result is.
This example proposes a equivalent model with no discrepancies.

.. contents::
    :local:

Imports, setups
+++++++++++++++

All imports. It also registered onnx converters for :epgk:`xgboost`
and :epkg:`lightgbm`.
"""
import pprint
import numpy
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from onnxruntime import InferenceSession
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnxrt import OnnxInference


def print_sparse_matrix(m):
    nonan = numpy.nan_to_num(m)
    mi, ma = nonan.min(), nonan.max()
    if mi == ma:
        ma += 1
    mat = numpy.empty(m.shape, dtype=numpy.str_)
    mat[:, :] = '.'
    if hasattr(m, 'todense'):
        dense = m.todense()
    else:
        dense = m
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if dense[i, j] > 0:
                c = int((dense[i, j] - mi) / (ma - mi) * 25)
                mat[i, j] = chr(ord('A') + c)
    return '\n'.join(''.join(line) for line in mat)


def max_diff(a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"Cannot compare matrices with different shapes "
            f"{a.shape} != {b.shape}.")
    d = numpy.abs(a - b).max()
    return d

#%%
# Artificial datasets
# +++++++++++++++++++
#
# Iris + a text column.


strings = numpy.array([
    "This a sentence.",
    "This a sentence with more characters $^*&'(-...",
    """var = ClassName(var2, user=mail@anywhere.com, pwd"""
    """=")_~-('&]@^\\`|[{#")""",
    "c79857654",
    "https://complex-url.com/;76543u3456?g=hhh&amp;h=23",
    "This is a kind of timestamp 01-03-05T11:12:13",
    "https://complex-url.com/;dd76543u3456?g=ddhhh&amp;h=23",
]).reshape((-1, 1))
labels = numpy.array(['http' in s for s in strings[:, 0]], dtype=numpy.int64)

pprint.pprint(strings)

#%%
# Fit a TfIdfVectorizer
# +++++++++++++++++++++

tfidf = Pipeline([
    ('pre', ColumnTransformer([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2)), 0)
    ]))
])

#%%
# We leave a couple of strings out of the training set.

tfidf.fit(strings[:-2])
tr = tfidf.transform(strings)
tfidf_step = tfidf.steps[0][1].transformers_[0][1]
pprint.pprint(f"output columns: {tfidf_step.get_feature_names_out()}")
print(f"rendered outputs, shape={tr.shape!r}")
print(print_sparse_matrix(tr))

#%%
# Conversion to ONNX
# ++++++++++++++++++

onx = to_onnx(tfidf, strings)
print(onnx_simple_text_plot(onx))


#%%
# Execution with ONNX and explanation of the discrepancies
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for rt in ['python', 'onnxruntime1']:
    oinf = OnnxInference(onx, runtime=rt)
    got = oinf.run({'X': strings})['variable']
    print(f"runtime={rt!r}, shape={got.shape!r}, "
          f"differences={max_diff(tr, got):g}")
    print(print_sparse_matrix(got))

#%%
# The conversion to ONNX is not exactly the same. The Tokenizer
# produces differences. By looking at the tokenized strings by onnx,
# word `h` appears in sequence `amp|h|23` and the bi-grams `amp,23`
# is never produced on this short example.

oinf = OnnxInference(onx, runtime='python', inplace=False)
res = oinf.run({'X': strings}, intermediate=True)
pprint.pprint(list(map(lambda s: '|'.join(s), res['tokenized'])))

#%%
# By default, :epkg:`scikit-learn` uses a regular expression.

print(f"tokenizer pattern: {tfidf_step.token_pattern!r}.")

#%%
# :epkg:`onnxruntime` uses :epkg:`re2` to handle the regular expression
# and there are differences with python regular expressions.

onx = to_onnx(tfidf, strings,
              options={TfidfVectorizer: {'tokenexp': r'(?u)\b\w\w+\b'}})
print(onnx_simple_text_plot(onx))
try:
    InferenceSession(onx.SerializeToString())
except Exception as e:
    print(f"ERROR: {e!r}.")

#%%
# A pipeline
# ++++++++++
#
# Let's assume the pipeline is followed by a logistic regression.

pipe = Pipeline([
    ('pre', ColumnTransformer([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2)), 0)])),
    ('logreg', LogisticRegression())])
pipe.fit(strings[:-2], labels[:-2])
pred = pipe.predict_proba(strings)
print(f"predictions:\n{pred}")

#%%
# Let's convert into ONNX and check the predictions.

onx = to_onnx(pipe, strings, options={'zipmap': False})
for rt in ['python', 'onnxruntime1']:
    oinf = OnnxInference(onx, runtime=rt)
    pred_onx = oinf.run({'X': strings})['probabilities']
    d = max_diff(pred, pred_onx)
    print(f"ONNX prediction {rt!r} - diff={d}:\n{pred_onx!r}")

#%%
# There are discrepancies introduces by the fact the regular expression
# uses in ONNX and by scikit-learn are not exactly the same.
# In this case, the runtime cannot replicate what python does.
# The runtime can be changed (see :epkg:`onnxruntime-extensions`).
# This example explores another direction.
#
# Replace the TfIdfVectorizer by ONNX before next step
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# 

