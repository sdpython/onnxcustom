"""
Implement a new converter
=========================

By default, :epkg:`sklearn-onnx` assumes that a classifier
has two outputs (label and probabilities), a regressor
has one output (prediction), a transform has one output
(the transformed data). This example assumes the model to
convert is one of them. In that case, a new converter requires
in fact two functions:

* a shape calculator: it defines the output shape and type
  based on the model and input type,
* a converter: it actually builds an ONNX graph equivalent
  to the prediction function to be converted.

This example implements both components for a new model.

.. contents::
    :local:

Custom model
++++++++++++

Let's implement a simple custom model using
:epkg:`scikit-learn` API. The model is preprocessing
which decorrelates correlated random variables.
If *X* is a matrix of features, :math:`V=\frac{1}{n}X'X`
is the covariance matrix. We compute :math:`X V^{1/2}`.
"""
import numpy
from sklearn.base import TransformerMixin, BaseEstimator


class DecorrelateTransformer(TransformerMixin, BaseEstimator):
    """
    Decorrelates correlated gaussiance features.

    :param alpha: avoids non inversible matrices
    """

    def __init__(self, alpha=0.):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.alpha = alpha

    def fit(self, X, y=None, sample_weights=None):
        if sample_weights is not None:
            raise NotImplementedError(
                "sample_weights != None is not implemented.")
        V = X.T @ X / X.shape[0]
        if self.alpha != 0:
            V += numpy.identity(V.shape[0]) * self.alpha
        L, P = numpy.linalg.eig(V)
        L = L ** -0.5
        root = P @ numpy.diag(L) @ P.transpose()
        self.coef_ = root ** (-1)
        return self

    def transform(self, X):
        return X @ self.coef_
