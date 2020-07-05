"""
A new converter with options
============================

Options are used to implement different conversion
for a same model. The options can be used to replace
an operator by the Einsum operator and compare the
processing time for both graph. Let's see how to retrieve
the options within a converter.

Both examples :ref:`l-plot-custom-converter` and
:ref:`l-plot-custom-converter-wrapper` show two
transformers which does similar thing. Let's take the
transformer from :ref:`l-plot-custom-converter-wrapper`
and select with an option the way it should be converted.

.. contents::
    :local:

Custom model
++++++++++++

"""
import numpy
from onnxruntime import InferenceSession
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.algebra.onnx_ops import (
    OnnxIdentity, OnnxSub, OnnxDiv, OnnxMatMul)
from skl2onnx.algebra.onnx_operator import OnnxSubOperator
from skl2onnx import to_onnx


class DecorrelateTransformer(TransformerMixin, BaseEstimator):
    """
    Decorrelates correlated gaussiance features.

    :param alpha: avoids non inversible matrices

    *Attributes*

    * `self.mean_`: average
    * `self.coef_`: square root of the coveriance matrix
    """

    def __init__(self, alpha=0.):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.alpha = alpha

    def fit(self, X, y=None, sample_weights=None):
        self.pca_ = PCA(X.shape[1])
        self.pca_.fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


data = load_iris()
X = data.data

dec = DecorrelateTransformer()
dec.fit(X)
pred = dec.transform(X[:5])
print(pred)


############################################
# Conversion into ONNX
# ++++++++++++++++++++
#
# Let's try to convert it to see what happens.


def decorrelate_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    output_type = input_type([input_dim, op.pca_.components_.shape[1]])
    operator.outputs[0].type = output_type


def decorrelate_transformer_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    # We retrieve the unique input.
    X = operator.inputs[0]

    options = container.get_options(op, dict(use_pca=True))
    use_pca = options['use_pca']
    dtype = guess_numpy_type(X.type)

    print("use_pca", use_pca)

    if use_pca:
        subop = OnnxSubOperator(op.pca_, X, op_version=opv)
        Y = OnnxIdentity(subop, op_version=opv, output_names=out[:1])
    else:
        center = OnnxSub(X, op.pca_.mean_.astype(dtype),
                         op_version=opv)
        tr = OnnxMatMul(center, op.pca_.components_.T.astype(dtype),
                        op_version=opv)

        if op.pca_.whiten:
            Y = OnnxDiv(
                tr, numpy.sqrt(op.pca_.explained_variance_).astype(dtype),
                op_version=opv, output_names=out[:1])
        else:
            Y = OnnxIdentity(tr, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


###################################
# The registration needs to declare the options
# supported by the converted.


update_registered_converter(
    DecorrelateTransformer, "SklearnDecorrelateTransformer",
    decorrelate_transformer_shape_calculator,
    decorrelate_transformer_converter,
    options={'use_pca': [True, False]})


onx = to_onnx(dec, X.astype(numpy.float32))

sess = InferenceSession(onx.SerializeToString())

exp = dec.transform(X.astype(numpy.float32))
got = sess.run(None, {'X': X.astype(numpy.float32)})[0]


def diff(p1, p2):
    p1 = p1.ravel()
    p2 = p2.ravel()
    d = numpy.abs(p2 - p1)
    return d.max(), (d / numpy.abs(p1)).max()


print(diff(exp, got))

############################################
# We try the non default option, use_pca: False.

onx = to_onnx(dec, X.astype(numpy.float32),
              options={'use_pca': False})

sess = InferenceSession(onx.SerializeToString())

exp = dec.transform(X.astype(numpy.float32))
got = sess.run(None, {'X': X.astype(numpy.float32)})[0]

print(diff(exp, got))
