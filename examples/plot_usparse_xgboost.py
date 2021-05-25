"""
.. _example-sparse-tfidf:

TfIdf and sparse matrices
=========================

.. index:: XGBoost, lightgbm, RandomForest


.. contents::
    :local:

Train a RandomForestClassifier after sparse
+++++++++++++++++++++++++++++++++++++++++++
"""
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference
import numpy
import pandas
import onnxruntime as rt
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost)
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
    convert_lightgbm)


update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
update_registered_converter(
    LGBMClassifier, 'LightGbmLGBMClassifier',
    calculate_linear_classifier_output_shapes, convert_lightgbm,
    options={'nocl': [True, False], 'zipmap': [True, False]})


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


pipe = Pipeline([
    ('union', ColumnTransformer([
        ('scale1', StandardScaler(), [0, 1]),
        ('subject',
         Pipeline([
             ('count', CountVectorizer()),
             ('tfidf', TfidfTransformer())
         ]), "text"),
    ], sparse_threshold=1.)),
    ('cls', RandomForestClassifier(n_estimators=5, max_depth=3)),
])

pipe.fit(df, y)


# Convert

model_onnx = convert_sklearn(
    pipe, 'pipeline_xgboost',
    [('input', FloatTensorType([None, 2])),
     ('text', StringTensorType([None, 1]))],
    target_opset=12,
    options={RandomForestClassifier: {'zipmap': False}})


# Compare the predictions

print("predict", pipe.predict(df[:5]))
print("predict_proba", pipe.predict_proba(df[:2]))

# Predictions with onnxruntime.

sess = rt.InferenceSession(model_onnx.SerializeToString())
pred_onx = sess.run(None, {
    "input": df[[0, 1]].values.astype(numpy.float32),
    "text": df[["text"]].values})
print("predict", pred_onx[0][:5])
print("predict_proba", pred_onx[1][:2])

print("%s differences:" % pipe.steps[-1][-1].__class__.__name__,
      numpy.abs(pred_onx[1].ravel() - pipe.predict_proba(df).ravel()).sum())

############################################
# Train a XGBoost after sparse
# ++++++++++++++++++++++++++++

pipe = Pipeline([
    ('union', ColumnTransformer([
        ('scale1', StandardScaler(), [0, 1]),
        ('subject',
         Pipeline([
             ('count', CountVectorizer(ngram_range=(1, 2))),
             ('tfidf', TfidfTransformer())
         ]), "text"),
    ], sparse_threshold=1.)),
    ('cls', XGBClassifier(n_estimators=5, max_depth=3)),
])

pipe.fit(df, y)

model_onnx = convert_sklearn(
    pipe, 'pipeline_xgboost',
    [('input', FloatTensorType([None, 2])),
     ('text', StringTensorType([None, 1]))],
    target_opset=12,
    options={XGBClassifier: {'zipmap': False}})

print("predict", pipe.predict(df[:5]))
print("predict_proba", pipe.predict_proba(df[:2]))

with open('model.onnx', 'wb') as f:
    f.write(model_onnx.SerializeToString())

sess = rt.InferenceSession(model_onnx.SerializeToString())
pred_onx = sess.run(None, {
    "input": df[[0, 1]].values.astype(numpy.float32),
    "text": df[["text"]].values})
print("predict", pred_onx[0][:5])
print("predict_proba", pred_onx[1][:2])

print("%s differences:" % pipe.steps[-1][-1].__class__.__name__,
      numpy.abs(pred_onx[1].ravel() - pipe.predict_proba(df).ravel()).sum())


############################################
# Train a LightGBM after sparse
# +++++++++++++++++++++++++++++

pipe = Pipeline([
    ('union', ColumnTransformer([
        ('scale1', StandardScaler(), [0, 1]),
        ('subject',
         Pipeline([
             ('count', CountVectorizer(ngram_range=(1, 2))),
             ('tfidf', TfidfTransformer())
         ]), "text"),
    ], sparse_threshold=1.)),
    ('cls', LGBMClassifier(n_estimators=5, max_depth=3)),
])

pipe.fit(df, y)

model_onnx = convert_sklearn(
    pipe, 'pipeline_lgb',
    [('input', FloatTensorType([None, 2])),
     ('text', StringTensorType([None, 1]))],
    target_opset=12,
    options={LGBMClassifier: {'zipmap': False}})

print("predict", pipe.predict(df[:5]))
print("predict_proba", pipe.predict_proba(df[:2]))

with open('model.onnx', 'wb') as f:
    f.write(model_onnx.SerializeToString())

sess = rt.InferenceSession(model_onnx.SerializeToString())
pred_onx = sess.run(None, {
    "input": df[[0, 1]].values.astype(numpy.float32),
    "text": df[["text"]].values})
print("predict", pred_onx[0][:5])
print("predict_proba", pred_onx[1][:2])

print("%s differences:" % pipe.steps[-1][-1].__class__.__name__,
      numpy.abs(pred_onx[1].ravel() - pipe.predict_proba(df).ravel()).sum())


#############################
# Final graph
# +++++++++++


oinf = OnnxInference(model_onnx)
ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
