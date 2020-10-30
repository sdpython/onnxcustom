"""
Add a parser to handle dataframes
=================================

.. index:: parser, dataframe

:ref:`l-custom-parser` shows how to add a parser to define
a converter a model which works differently than standard
predictors of :epkg:`scikit-learn`. In this case,
the input is a dataframe and takes an input per column
of the dataframe. One input is impossible because a dataframe
may contain different types.

.. contents::
    :local:

A transformer taking a dataframe as input
+++++++++++++++++++++++++++++++++++++++++
"""
from pprint import pprint
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference
import numpy
from pandas import DataFrame
from onnxruntime import InferenceSession
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OrdinalEncoder
from skl2onnx import update_registered_converter, to_onnx
from skl2onnx._parse import _parse_sklearn_simple_model
from skl2onnx.common.data_types import (
    Int64TensorType, StringTensorType, FloatTensorType,
    _guess_numpy_type)


class DiscretizeTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, thresholds):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.thresholds = thresholds

    def fit(self, X, y=None, sample_weights=None):
        # Does nothing.
        if len(X.shape) != 2 or X.shape[1]:
            raise RuntimeError("The transformer expects only one columns.")
        return self

    def transform(self, X):
        return numpy.digitize(X, self.thresholds).reshape((-1, 1))


class PreprocessDataframeTransformer(TransformerMixin, BaseEstimator):
    """
    Converts all columns of a dataframe in integers
    than in floats.
    """

    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

    def fit(self, X, y=None, sample_weights=None):
        "Trains the transformer. Creates the member `args_`."
        if sample_weights is not None:
            raise NotImplementedError(
                "sample_weights != None is not implemented.")
        if not isinstance(X, DataFrame):
            raise TypeError("X must be a dataframe.")
        self.args_ = []
        for i, (col, dt) in enumerate(zip(X.columns, X.dtypes)):
            values = X[col].values
            if dt in (numpy.float32, numpy.float64):
                qu = numpy.quantile(values, numpy.arange(4) * 0.25)
                self.args_.append((i, col, dt, DiscretizeTransformer(qu)))
            elif dt == 'category':
                oo = OrdinalEncoder(dtype=numpy.int64)
                values = values.to_numpy()
                oo.fit(values.reshape((-1, 1)))
                self.args_.append((i, col, dt, oo))
            else:
                raise RuntimeError(
                    "Unable to transform column '{}' type: '{}'.".format(
                        col, dt))
        return self

    def transform(self, X):
        if not isinstance(X, DataFrame):
            raise TypeError("X must be a dataframe.")
        outs = []
        for i, col, dt, arg in self.args_:
            if X.columns[i] != col:
                raise RuntimeError(
                    "Unexpected column name '{}' at position {}.".format(
                        col, i))
            if X.dtypes[i] != dt:
                raise RuntimeError(
                    "Unexpected column type '{}' at position {}.".format(
                        col, i))
            values = X[col].values
            if dt in (numpy.float32, numpy.float64):
                out = arg.transform(values)
            elif dt == 'category':
                values = values.to_numpy()
                out = arg.transform(values.reshape((-1, 1)))
            outs.append(out)
        res = numpy.hstack(outs)
        return res.astype(numpy.float32)


data = DataFrame([
    dict(afloat=0.5, anint=4, astring="A"),
    dict(afloat=0.6, anint=5, astring="B"),
    dict(afloat=0.7, anint=6, astring="C"),
    dict(afloat=0.8, anint=5, astring="D"),
    dict(afloat=0.9, anint=4, astring="C"),
    dict(afloat=1.0, anint=5, astring="B")])
data['afloat'] = data['afloat'].astype(numpy.float32)
data['anint'] = data['anint'].astype('category')
data['astring'] = data['astring'].astype('category')

dec = PreprocessDataframeTransformer()
dec.fit(data)
pred = dec.transform(data)
print(pred)


############################################
# Conversion into ONNX
# ++++++++++++++++++++
#
# The transform has multiple inputs but one outputs.
# This case is not standard and requires a custom parser.
# The model ingests different types but returns one output.


def preprocess_dataframe_transformer_parser(
        scope, model, inputs, custom_parsers=None):
    if len(inputs) != len(model.args_):
        raise RuntimeError(
            "Converter expects {} inputs but got {}.".format(
                len(model.args_), len(inputs)))
    transformed_result_names = []
    for i, col, dt, arg in model.args_:
        if dt in (numpy.float32, numpy.float64):
            op = scope.declare_local_operator('CustomDiscretizeTransformer')
            op.inputs = [inputs[i]]
            op.raw_operator = arg
            op_var = scope.declare_local_variable(
                'output{}'.format(i), Int64TensorType())
            op.outputs.append(op_var)
            transformed_result_names.append(op.outputs[0])
        elif dt == 'category':
            transformed_result_names.append(
                _parse_sklearn_simple_model(
                    scope, arg, [inputs[i]],
                    custom_parsers=custom_parsers)[0])

    # Create a Concat ONNX node
    concat_operator = scope.declare_local_operator('SklearnConcat')
    concat_operator.inputs = transformed_result_names
    union_name = scope.declare_local_variable(
        'union', FloatTensorType())
    concat_operator.outputs.append(union_name)
    return concat_operator.outputs


def preprocess_dataframe_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_dim = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([input_dim, len(op.args_)])


def preprocess_dataframe_transformer_converter(scope, operator, container):
    # op = operator.raw_operator
    # opv = container.target_opset
    # out = operator.outputs
    raise NotImplementedError(
        "Converter for PreprocessDataframeTransformer is "
        "implemented in the parser.")


update_registered_converter(
    PreprocessDataframeTransformer,
    "CustomPreprocessDataframeTransformer",
    preprocess_dataframe_transformer_shape_calculator,
    preprocess_dataframe_transformer_converter,
    parser=preprocess_dataframe_transformer_parser)


#############################################
# And conversion.

def guess_schema_from_data(data):
    res = []
    for col, dt in zip(data.columns, data.dtypes):
        try:
            is_cat = dt == 'category'
        except TypeError:
            is_cat = False
        if is_cat:
            if isinstance(dt.categories[0], str):
                res.append((col, StringTensorType([None, 1])))
            else:
                res.append((col, Int64TensorType([None, 1])))
        else:
            res.append((col, _guess_numpy_type(dt, [None, 1])))
    return res


initial_types = guess_schema_from_data(data)
print(initial_types)

try:
    onx = to_onnx(dec, initial_types=initial_types, target_opset=12)
except RuntimeError as e:
    print(e)


##############################################
# The converter for alias DiscretizeTransform is not here.
# Let's add it.


def discretizer_transformer_shape_calculator(operator):
    operator.outputs[0].type = operator.inputs[0].type.__class__([None, 1])


def discretizer_transformer_converter(scope, operator, container):
    op = operator.raw_operator

    th = op.thresholds
    nth = th.shape[0]
    if nth != 4:
        raise RuntimeError(
            "The following cod is written only for four bins.")

    th = [th[1], th[0], th[2], th[3]]  # bin4, bin5, bin6, bin7

    attrs = dict(
        aggregate_function='SUM',
        base_values=[0.],
        n_targets=1,
        nodes_featureids=[0 for i in range(nth * 2)],
        nodes_hitrates=[0. for i in range(nth * 2)],
        # nodes_missing_value_tracks_true=
        nodes_modes=(['BRANCH_LEQ' for i in range(nth)] +
                     ['LEAF' for i in range(nth)]),
        nodes_treeids=[0 for i in range(nth * 2)],
        nodes_nodeids=numpy.arange(8).tolist(),
        nodes_falsenodeids=[1, 4, 5, 6, 0, 0, 0, 0],
        nodes_truenodeids=[2, 5, 3, 7, 0, 0, 0, 0],
        nodes_values=th + [0. for i in range(nth)],
        # post_transform='NONE',
        target_ids=[0 for i in range(nth)],
        target_nodeids=(numpy.arange(4) + 4).astype(int).tolist(),
        target_treeids=[0 for i in range(nth)],
        target_weights=numpy.arange(nth).astype(numpy.float32).tolist())

    container.add_node(
        'TreeEnsembleRegressor',
        operator.inputs[0].full_name, operator.outputs[0].full_name,
        name=scope.get_unique_operator_name('TreeEnsembleRegressor'),
        op_domain='ai.onnx.ml', op_version=1, **attrs)


update_registered_converter(
    DiscretizeTransformer,
    "CustomDiscretizeTransformer",
    discretizer_transformer_shape_calculator,
    discretizer_transformer_converter)


initial_types = guess_schema_from_data(data)
pprint(initial_types)
onx = to_onnx(dec, initial_types=initial_types, target_opset=12)
sess = InferenceSession(onx.SerializeToString())


def cvt_col(values):
    if hasattr(values, 'to_numpy'):
        values = values.to_numpy()
    return values.reshape((-1, 1))


inputs = {c: cvt_col(data[c]) for c in data.columns}

exp = dec.transform(data)
results = sess.run(None, inputs)
y = results[0]


def diff(p1, p2):
    p1 = p1.ravel()
    p2 = p2.ravel()
    d = numpy.abs(p2 - p1)
    return d.max()


print("expected")
print(exp)
print("ONNX")
print(y)
print("difference", diff(exp, y))


#############################
# Final graph
# +++++++++++

oinf = OnnxInference(onx, runtime="python_compiled")
ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
