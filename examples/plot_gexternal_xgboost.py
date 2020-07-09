# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _example-xgboost:

Convert a pipeline with a XGBoost model
========================================

.. index:: XGBoost

:epkg:`sklearn-onnx` only converts :epkg:`scikit-learn` models
into :epkg:`ONNX` but many libraries implement :epkg:`scikit-learn`
API so that their models can be included in a :epkg:`scikit-learn`
pipeline. This example considers a pipeline including a :epkg:`XGBoost`
model. :epkg:`sklearn-onnx` can convert the whole pipeline as long as
it knows the converter associated to a *XGBClassifier*. Let's see
how to do it.

.. contents::
    :local:

Train a XGBoost classifier
++++++++++++++++++++++++++
"""
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference
import numpy
import onnxruntime as rt
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost)

data = load_iris()
X = data.data[:, :2]
y = data.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

pipe = Pipeline([('scaler', StandardScaler()),
                 ('lgbm', XGBClassifier(n_estimators=3))])
pipe.fit(X, y)

# The conversion fails but it is expected.

try:
    convert_sklearn(pipe, 'pipeline_xgboost',
                    [('input', FloatTensorType([None, 2]))],
                    target_opset=12)
except Exception as e:
    print(e)

# The error message tells no converter was found
# for :epkg:`XGBoost` models. By default, :epkg:`sklearn-onnx`
# only handles models from :epkg:`scikit-learn` but it can
# be extended to every model following :epkg:`scikit-learn`
# API as long as the module knows there exists a converter
# for every model used in a pipeline. That's why
# we need to register a converter.

######################################
# Register the converter for XGBClassifier
# ++++++++++++++++++++++++++++++++++++++++
#
# The converter is implemented in :epkg:`onnxmltools`:
# `onnxmltools...XGBoost.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# xgboost/operator_converters/XGBoost.py>`_.
# and the shape calculator:
# `onnxmltools...Classifier.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# xgboost/shape_calculators/Classifier.py>`_.

update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False]})

##################################
# Convert again
# +++++++++++++

model_onnx = convert_sklearn(
    pipe, 'pipeline_xgboost',
    [('input', FloatTensorType([None, 2]))],
    target_opset=12)

# And save.
with open("pipeline_xgboost.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

###########################
# Compare the predictions
# +++++++++++++++++++++++
#
# Predictions with XGBoost.

print("predict", pipe.predict(X[:5]))
print("predict_proba", pipe.predict_proba(X[:1]))

##########################
# Predictions with onnxruntime.

sess = rt.InferenceSession("pipeline_xgboost.onnx")
pred_onx = sess.run(None, {"input": X[:5].astype(numpy.float32)})
print("predict", pred_onx[0])
print("predict_proba", pred_onx[1][:1])

#############################
# Final graph
# +++++++++++


oinf = OnnxInference(model_onnx)
ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
