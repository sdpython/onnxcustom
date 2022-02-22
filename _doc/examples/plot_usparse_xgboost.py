"""
.. _example-sparse-tfidf:

TfIdf and sparse matrices
=========================

.. index:: xgboost, lightgbm, sparse, ensemble

`TfidfVectorizer <https://scikit-learn.org/stable/modules/
generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_
usually creates sparse data. If the data is sparse enough, matrices
usually stays as sparse all along the pipeline until the predictor
is trained. Sparse matrices do not consider null and missing values
as they are not present in the datasets. Because some predictors
do the difference, this ambiguity may introduces discrepencies
when converter into ONNX. This example looks into several configurations.

.. contents::
    :local:

Imports, setups
+++++++++++++++

All imports. It also registered onnx converters for :epgk:`xgboost`
and :epkg:`lightgbm`.
"""
import warnings
import numpy
import pandas
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.experimental import (  # noqa
    enable_hist_gradient_boosting)  # noqa
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.sklapi import CastTransformer, ReplaceTransformer
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost)
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
    convert_lightgbm)
from mlprodict.onnxrt import OnnxInference


update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
update_registered_converter(
    LGBMClassifier, 'LightGbmLGBMClassifier',
    calculate_linear_classifier_output_shapes, convert_lightgbm,
    options={'nocl': [True, False], 'zipmap': [True, False]})


##########################################
# Artificial datasets
# +++++++++++++++++++++++++++
#
# Iris + a text column.

cst = ['class zero', 'class one', 'class two']

data = load_iris()
X = data.data[:, :2]
y = data.target

df = pandas.DataFrame(X)
df["text"] = [cst[i] for i in y]


ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()


##########################################
# Train ensemble after sparse
# +++++++++++++++++++++++++++
#
# The example use the Iris datasets with artifical text datasets
# preprocessed with a tf-idf. `sparse_threshold=1.` avoids
# sparse matrices to be converted into dense matrices.


def make_pipelines(df_train, y_train, models=None,
                   sparse_threshold=1., replace_nan=False,
                   insert_replace=False, verbose=False):

    if models is None:
        models = [
            RandomForestClassifier, HistGradientBoostingClassifier,
            XGBClassifier, LGBMClassifier]
    models = [_ for _ in models if _ is not None]

    pipes = []
    for model in tqdm(models):

        if model == HistGradientBoostingClassifier:
            kwargs = dict(max_iter=5)
        elif model == XGBClassifier:
            kwargs = dict(n_estimators=5, use_label_encoder=False)
        else:
            kwargs = dict(n_estimators=5)

        if insert_replace:
            pipe = Pipeline([
                ('union', ColumnTransformer([
                    ('scale1', StandardScaler(), [0, 1]),
                    ('subject',
                     Pipeline([
                         ('count', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('repl', ReplaceTransformer()),
                     ]), "text"),
                ], sparse_threshold=sparse_threshold)),
                ('cast', CastTransformer()),
                ('cls', model(max_depth=3, **kwargs)),
            ])
        else:
            pipe = Pipeline([
                ('union', ColumnTransformer([
                    ('scale1', StandardScaler(), [0, 1]),
                    ('subject',
                     Pipeline([
                         ('count', CountVectorizer()),
                         ('tfidf', TfidfTransformer())
                     ]), "text"),
                ], sparse_threshold=sparse_threshold)),
                ('cast', CastTransformer()),
                ('cls', model(max_depth=3, **kwargs)),
            ])

        try:
            pipe.fit(df_train, y_train)
        except TypeError as e:
            obs = dict(model=model.__name__, pipe=pipe, error=e)
            pipes.append(obs)
            continue

        options = {model: {'zipmap': False}}
        if replace_nan:
            options[TfidfTransformer] = {'nan': True}

        # convert
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore", (FutureWarning, UserWarning))
            model_onnx = to_onnx(
                pipe,
                initial_types=[('input', FloatTensorType([None, 2])),
                               ('text', StringTensorType([None, 1]))],
                target_opset={'': 14, 'ai.onnx.ml': 2},
                options=options)

        with open('model.onnx', 'wb') as f:
            f.write(model_onnx.SerializeToString())

        oinf = OnnxInference(model_onnx)
        inputs = {"input": df[[0, 1]].values.astype(numpy.float32),
                  "text": df[["text"]].values}
        pred_onx = oinf.run(inputs)

        diff = numpy.abs(
            pred_onx['probabilities'].ravel() -
            pipe.predict_proba(df).ravel()).sum()

        if verbose:
            def td(a):
                if hasattr(a, 'todense'):
                    b = a.todense()
                    ind = set(a.indices)
                    for i in range(b.shape[1]):
                        if i not in ind:
                            b[0, i] = numpy.nan
                    return b
                return a

            oinf = OnnxInference(model_onnx)
            pred_onx2 = oinf.run(inputs)
            diff2 = numpy.abs(
                pred_onx2['probabilities'].ravel() -
                pipe.predict_proba(df).ravel()).sum()

        if diff > 0.1:
            for i, (l1, l2) in enumerate(
                    zip(pipe.predict_proba(df),
                        pred_onx['probabilities'])):
                d = numpy.abs(l1 - l2).sum()
                if verbose and d > 0.1:
                    print("\nDISCREPENCY DETAILS")
                    print(d, i, l1, l2)
                    pre = pipe.steps[0][-1].transform(df)
                    print("idf", pre[i].dtype, td(pre[i]))
                    pre2 = pipe.steps[1][-1].transform(pre)
                    print("cas", pre2[i].dtype, td(pre2[i]))
                    inter = oinf.run(inputs, intermediate=True)
                    onx = inter['tfidftr_norm']
                    print("onx", onx.dtype, onx[i])
                    onx = inter['variable3']

        obs = dict(model=model.__name__,
                   discrepencies=diff,
                   model_onnx=model_onnx, pipe=pipe)
        if verbose:
            obs['discrepency2'] = diff2
        pipes.append(obs)

    return pipes


data_sparse = make_pipelines(df, y)
stat = pandas.DataFrame(data_sparse).drop(['model_onnx', 'pipe'], axis=1)
if 'error' in stat.columns:
    print(stat.drop('error', axis=1))
stat

############################
# Sparse data hurts.
#
# Dense data
# ++++++++++
#
# Let's replace sparse data with dense by using `sparse_threshold=0.`


data_dense = make_pipelines(df, y, sparse_threshold=0.)
stat = pandas.DataFrame(data_dense).drop(['model_onnx', 'pipe'], axis=1)
if 'error' in stat.columns:
    print(stat.drop('error', axis=1))
stat

####################################
# This is much better. Let's compare how the preprocessing
# applies on the data.

print("sparse")
print(data_sparse[-1]['pipe'].steps[0][-1].transform(df)[:2])
print()
print("dense")
print(data_dense[-1]['pipe'].steps[0][-1].transform(df)[:2])

####################################
# This shows `RandomForestClassifier
# <https://scikit-learn.org/stable/modules/generated/
# sklearn.ensemble.RandomForestClassifier.html>`_,
# `XGBClassifier <https://xgboost.readthedocs.io/
# en/latest/python/python_api.html>`_ do not process
# the same way sparse and
# dense matrix as opposed to `LGBMClassifier
# <https://lightgbm.readthedocs.io/en/latest/
# pythonapi/lightgbm.LGBMClassifier.html>`_.
# And `HistGradientBoostingClassifier
# <https://scikit-learn.org/stable/modules/generated/
# sklearn.ensemble.HistGradientBoostingClassifier.html>`_
# fails.
#
# Dense data with nan
# +++++++++++++++++++
#
# Let's keep sparse data in the scikit-learn pipeline but
# replace null values by nan in the onnx graph.

data_dense = make_pipelines(df, y, sparse_threshold=1., replace_nan=True)
stat = pandas.DataFrame(data_dense).drop(['model_onnx', 'pipe'], axis=1)
if 'error' in stat.columns:
    print(stat.drop('error', axis=1))
stat


##############################
# Dense, 0 replaced by nan
# ++++++++++++++++++++++++
#
# Instead of using a specific options to replace null values
# into nan values, a custom transformer called
# ReplaceTransformer is explicitely inserted into the pipeline.
# A new converter is added to the list of supported models.
# It is equivalent to the previous options except it is
# more explicit.

data_dense = make_pipelines(df, y, sparse_threshold=1., replace_nan=False,
                            insert_replace=True)
stat = pandas.DataFrame(data_dense).drop(['model_onnx', 'pipe'], axis=1)
if 'error' in stat.columns:
    print(stat.drop('error', axis=1))
stat

######################################
# Conclusion
# ++++++++++
#
# Unless dense arrays are used, because :epkg:`onnxruntime`
# ONNX does not support sparse yet, the conversion needs to be
# tuned depending on the model which follows the TfIdf preprocessing.
