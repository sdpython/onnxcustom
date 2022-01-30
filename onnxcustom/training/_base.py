"""
@file
@brief Base class for @see cl BaseEstimator and @see cl BaseOnnxFunction.
"""
import os
import inspect
import warnings


class BaseOnnxClass:
    """
    Bases class with common functions to handle attributes
    in classes owning ONNX graphs.
    """

    @classmethod
    def _get_param_names(cls):
        "Extracts all parameters to serialize."
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_signature = inspect.signature(init)
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD]
        return [(p.name, p.default) for p in parameters]

    def save_onnx_graph(self, folder, prefix=None, suffix=None):
        """
        Saves all ONNX files stored in this class.

        :param folder: folder where to save (it must exists) or
            ``bytes`` if the onnx graph must be returned as bytes,
            not files
        :param prefix: suffix to add to the name
        :param suffix: suffix to add to the name
        :return: list of saved files (dictionary
            `{ attribute: filename or dictionary }`)

        The function raises a warning if a file already exists.
        The function uses class name, attribute names to compose
        file names. It shortens them for frequent classes.

        * 'Learning' -> 'L'
        * 'OrtGradient' -> 'Grad'
        * 'ForwardBackward' -> 'FB'

        .. runpython::
            :showcode:

            import io
            import numpy
            import onnx
            from sklearn.datasets import make_regression
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from skl2onnx import to_onnx
            from mlprodict.plotting.text_plot import onnx_simple_text_plot
            from onnxcustom.training.optimizers_partial import (
                OrtGradientForwardBackwardOptimizer)
            from onnxcustom.training.sgd_learning_rate import (
                LearningRateSGDNesterov)
            from onnxcustom.training.sgd_learning_penalty import (
                ElasticLearningPenalty)


            def walk_through(obj, prefix="", only_name=True):
                for k, v in obj.items():
                    if isinstance(v, dict):
                        p = prefix + "." + k if prefix else k
                        walk_through(v, prefix=p, only_name=only_name)
                    elif only_name:
                        name = "%s.%s" % (prefix, k) if prefix else k
                        print('+', name)
                    else:
                        name = "%s.%s" % (prefix, k) if prefix else k
                        print('\\n++++++', name)
                        print()
                        bf = io.BytesIO(v)
                        onx = onnx.load(bf)
                        print(onnx_simple_text_plot(onx))


            X, y = make_regression(  # pylint: disable=W0632
                100, n_features=3, bias=2, random_state=0)
            X = X.astype(numpy.float32)
            y = y.astype(numpy.float32)
            X_train, _, y_train, __ = train_test_split(X, y)
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            reg.coef_ = reg.coef_.reshape((1, -1))
            opset = 15
            onx = to_onnx(reg, X_train, target_opset=opset,
                          black_op={'LinearRegressor'})
            inits = ['coef', 'intercept']

            train_session = OrtGradientForwardBackwardOptimizer(
                onx, inits,
                learning_rate=LearningRateSGDNesterov(
                    1e-4, nesterov=False, momentum=0.9),
                learning_penalty=ElasticLearningPenalty(l1=1e-3, l2=1e-4),
                warm_start=False, max_iter=100, batch_size=10)

            onxs = train_session.save_onnx_graph(bytes)

            print("+ all onnx graphs")
            walk_through(onxs, only_name=True)
            walk_through(onxs, only_name=False)
        """
        repls = {'Learning': 'L', 'OrtGradient': 'Grad',
                 'ForwardBackward': 'FB'}
        if prefix is None:
            prefix = ''
        if suffix is None:
            suffix = ''
        if isinstance(folder, str) and not os.path.exists(folder):
            raise FileNotFoundError(  # pragma: no cover
                "Folder %r does not exist." % folder)
        saved = {}
        for k, v in self.__dict__.items():
            if hasattr(v, "SerializeToString"):
                if isinstance(folder, str):
                    name = "%s%s%s.%s.onnx" % (
                        prefix, self.__class__.__name__, suffix, k)
                    for a, b in repls.items():
                        name = name.replace(a, b)
                    filename = os.path.join(folder, name)
                    if os.path.exists(filename):
                        warnings.warn(  # pragma: no cover
                            "Filename %r already exists." % filename)
                    with open(filename, "wb") as f:
                        f.write(v.SerializeToString())
                    saved[k] = filename
                else:
                    saved[k] = v.SerializeToString()
            elif hasattr(v, "save_onnx_graph"):
                saved[k] = v.save_onnx_graph(
                    folder, prefix=prefix, suffix="%s.%s" % (suffix, k))
        return saved
