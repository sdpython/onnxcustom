"""
@brief      test log(time=11s)
"""
# pylint: disable=C0200,E1101,W0212
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from onnx import (  # pylint: disable=E0611
    ModelProto, TensorProto, numpy_helper)
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid)
from onnx.checker import check_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from onnxruntime import InferenceSession
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.utils.onnx_split import split_onnx, OnnxSplitting


class TestSplitOnnx(ExtTestCase):

    @ignore_warnings(ConvergenceWarning)
    def test_split_onnx(self):
        X = numpy.arange(30).reshape((-1, 3)).astype(numpy.float32) / 100
        y = numpy.arange(X.shape[0]).astype(numpy.float32)
        y = y.reshape((-1, 1))
        reg = MLPRegressor(hidden_layer_sizes=(5,), max_iter=2,
                           activation='logistic',
                           momentum=0, nesterovs_momentum=False,
                           alpha=0)
        reg.fit(X, y.ravel())

        onx = to_onnx(reg, X, target_opset=opset)
        split = OnnxSplitting(onx)
        segs = split.segments
        self.assertGreater(len(segs), 3)
        self.assertIn("OnnxSegment(", repr(segs[0]))
        size0 = len(onx.SerializeToString())
        total = 0
        for seg in split.segments:
            self.assertLess(seg.size, size0)
            total += seg.size
        self.assertLess(size0, total)

        def myprint(*args):
            pass

        for n_parts in [2, 4]:
            with self.subTest(n_parts=n_parts):
                parts, stats = split_onnx(
                    onx, n_parts, verbose=1, fLOG=myprint, stats=True)
                self.assertIsInstance(stats, dict)
                for k in ['cutting_points', 'extremities',
                          'segments', 'split']:
                    self.assertIn(k, stats)
                self.assertEqual(len(parts), n_parts)
                for i, p in enumerate(parts):
                    if len(p.graph.input) == 0:
                        raise AssertionError(f"No input in part {i}\n{p}")
                    if len(p.graph.output) == 0:
                        raise AssertionError(f"No output in part {i}\n{p}")
                    if len(p.graph.node) == 0:
                        raise AssertionError(f"No node in part {i}\n{p}")
                    self.assertIsInstance(p, ModelProto)

                main = InferenceSession(onx.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
                rtp = [InferenceSession(p.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
                       for p in parts]

                expected = reg.predict(X)
                got = main.run(None, {'X': X})[0]
                self.assertEqualArray(expected, numpy.squeeze(got), decimal=6)

                feeds = {'X': X}
                for rt in rtp:
                    out = rt.run(None, feeds)[0]
                    n = rt.get_outputs()[0].name
                    feeds = {n: out}
                self.assertEqualArray(expected, numpy.squeeze(out), decimal=6)

    def test_split_onnx_branch(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None, None])
        T = make_tensor_value_info('T', TensorProto.FLOAT, [None, None])
        nodes = [make_node('Sub', ['X', 'Y'], ['diff']),
                 make_node('Mul', ['diff', 'diff'], ['abs']),
                 make_node('Add', ['abs', 'Z'], ['dz1']),
                 make_node('Sub', ['abs', 'Z'], ['dz2']),
                 make_node('Mul', ['dz1', 'dz2'], ['T'])]

        graph = make_graph(nodes, "dummy", [X, Y, Z], [T])
        onx = make_model(graph)
        check_model(onx)

        split = OnnxSplitting(onx)
        self.assertNotIn("dx", split.cutting_points)
        self.assertNotIn("dy", split.cutting_points)
        self.assertIn("diff", split.cutting_points)
        parts = split_onnx(onx, 2)

        ids = set(n.SerializeToString() for n in onx.graph.node)
        total = 0
        for p in parts:
            keys = [n.SerializeToString() for n in p.graph.node]
            for k in keys:
                if k not in ids:
                    raise AssertionError("node not found")
            total += len(keys)
        self.assertEqual(len(onx.graph.node), total)

        for i in range(len(split.segments)):
            ox, _ = split._make_onnx(i, i + 1)
            self.assertNotEmpty(ox.graph.input)
            self.assertNotEmpty(ox.graph.output)

    @staticmethod
    def create_model(add=False):
        initializers = []
        nodes = []
        inputs = []
        outputs = []
        functions = []

        opsets = {'': 10}

        if add == 1:
            value = numpy.array([1e-5], dtype=numpy.float32)
            tensor = numpy_helper.from_array(value, name='IADD')
            initializers.append(tensor)

        value = numpy.random.randn(96, 16, 1, 1).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='I1')
        initializers.append(tensor)

        value = numpy.random.randn(96).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='I2')
        initializers.append(tensor)

        value = numpy.random.randn(96, 1, 3, 3).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='I3')
        initializers.append(tensor)

        value = numpy.random.randn(96).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='I4')
        initializers.append(tensor)

        value = numpy.random.randn(32, 96, 1, 1).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='I5')
        initializers.append(tensor)

        value = numpy.random.randn(32).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='I6')
        initializers.append(tensor)

        value = numpy.random.randn(128, 32, 1, 1).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='I7')
        initializers.append(tensor)

        value = numpy.random.randn(128).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='I8')
        initializers.append(tensor)

        tensor = numpy_helper.from_array(
            numpy.array([-1], dtype=numpy.int64), name='I9')
        initializers.append(tensor)

        value = numpy.random.randn(100).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='II1')

        initializers.append(tensor)

        value = numpy.random.randn(100, 128).astype(numpy.float32)
        tensor = numpy_helper.from_array(value, name='II2')

        initializers.append(tensor)

        inputs.append(make_tensor_value_info(
            'input', 1, ['batch_size', 3, 224, 224]))
        outputs.append(make_tensor_value_info(
            'output', 1, ['batch_size', 1000]))

        node = make_node(
            'Conv', ['input', 'I1', 'I2'], ['R1'],
            name='Conv1', dilations=[1, 1], group=1, kernel_shape=[1, 1],
            pads=[0, 0, 0, 0], strides=[1, 1])
        nodes.append(node)

        if add > 0:
            if add == 2:
                tensor = numpy_helper.from_array(
                    numpy.array([1e-5], dtype=numpy.float32), name='IADD')
                node = make_node('Constant', [], ['IADD'],
                                 name='CST1', value=tensor)
                nodes.append(node)

            node = make_node('Add', ['R1', 'IADD'], ['R1ADD'], name='Add1')
            nodes.append(node)

            node = make_node(
                'Clip', ['R1ADD'], ['R2'],
                name='Clip2', max=6.0, min=0.0, domain='')
            nodes.append(node)
        else:
            node = make_node(
                'Clip', ['R1'], ['R2'],
                name='Clip2', max=6.0, min=0.0, domain='')
            nodes.append(node)

        node = make_node(
            'Conv', ['R2', 'I3', 'I4'], ['R3'],
            name='Conv3', dilations=[1, 1], group=960,
            kernel_shape=[3, 3], pads=[1, 1, 1, 1],
            strides=[1, 1], domain='')
        nodes.append(node)

        node = make_node(
            'Clip', ['R3'], ['R4'],
            name='Clip4', max=6.0, min=0.0, domain='')
        nodes.append(node)

        node = make_node(
            'Conv', ['R4', 'I5', 'I6'], ['R5'],
            name='Conv5', dilations=[1, 1], group=1, kernel_shape=[1, 1],
            pads=[0, 0, 0, 0], strides=[1, 1], domain='')
        nodes.append(node)

        node = make_node(
            'Conv', ['R5', 'I7', 'I8'], ['R6'],
            name='Conv6', dilations=[1, 1], group=1, kernel_shape=[1, 1],
            pads=[0, 0, 0, 0], strides=[1, 1], domain='')
        nodes.append(node)

        node = make_node(
            'Clip', ['R6'], ['R7'],
            name='Clip7', max=6.0, min=0.0, domain='')
        nodes.append(node)

        node = make_node(
            'GlobalAveragePool', ['R7'], ['R8'],
            name='GlobalAveragePool8', domain='')
        nodes.append(node)

        node = make_node(
            'Shape', ['R7'], ['R9'],
            name='Shape9', domain='')
        nodes.append(node)

        node = make_node(
            'Constant', [], ['R10'],
            name='Constant10',
            value=make_tensor("value", TensorProto.INT64, dims=[1], vals=[0]),
            domain='')
        nodes.append(node)

        node = make_node(
            'Gather', ['R9', 'R10'], ['R11'],
            name='Gather11', axis=0, domain='')
        nodes.append(node)

        node = make_node(
            'Unsqueeze', ['R11'], ['R12'],
            name='Unsqueeze12', axes=[0], domain='')
        nodes.append(node)

        node = make_node(
            'Concat', ['R12', 'I9'], ['R13'],
            name='Concat13', axis=0, domain='')
        nodes.append(node)

        node = make_node(
            'Reshape', ['R8', 'R13'], ['R14'],
            name='Reshape14', domain='')
        nodes.append(node)

        if add > 0:
            node = make_node(
                'Gemm', ['R14', 'II2', 'II1'], ['R2ADD'],
                name='Gemm15', alpha=1.0, beta=1.0, transB=1, domain='')
            nodes.append(node)

            node = make_node('Add', ['R2ADD', 'IADD'], ['output'], name='Add2')
            nodes.append(node)
        else:
            node = make_node(
                'Gemm', ['R14', 'II2', 'II1'], ['output'],
                name='Gemm15', alpha=1.0, beta=1.0, transB=1, domain='')
            nodes.append(node)

        opset_imports = [make_opsetid(domain, 1 if version is None else version)
                         for domain, version in opsets.items()]

        graph = make_graph(nodes, 'torch-jit-export',
                           inputs, outputs, initializers)

        onnx_model = make_model(
            graph, opset_imports=opset_imports, functions=functions)
        onnx_model.ir_version = 6
        onnx_model.producer_name = 'pytorch'
        onnx_model.producer_version = ''
        onnx_model.domain = ''
        onnx_model.model_version = 0
        onnx_model.doc_string = ''
        set_model_props(onnx_model, {})
        check_model(onnx_model)
        return onnx_model

    def test_split_big_model(self):
        onx = self.create_model()

        split = OnnxSplitting(onx, verbose=0)
        self.assertNotIn('R10', split.cutting_points)
        self.assertIn('R7', split.cutting_points)
        for i in range(len(split.segments)):
            ox, _ = split._make_onnx(i, i + 1)
            self.assertNotEmpty(ox.graph.input)
            self.assertNotEmpty(ox.graph.output)
            if i > 0:
                self.assertEqual(
                    split.segments[i].begin, ox.graph.input[0].name)
            if i < len(split.segments) - 1:
                self.assertEqual(
                    split.segments[i].end, ox.graph.output[0].name)
            if ox.graph.input[0].name == 'R10' and ox.graph.node[0].op_type != 'Shape':
                with open("test_split_big_model.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
                raise AssertionError(
                    f"Unexpected segment {i}\n{split.segments[i]!r}"
                    f"\ninvolved={split.segments[i].involved}"
                    f"\n{onnx_simple_text_plot(ox)}")

        parts = split_onnx(onx, 2, verbose=0)

        ids = set(n.SerializeToString() for n in onx.graph.node)
        total = 0
        for p in parts:
            keys = [n.SerializeToString() for n in p.graph.node]
            for k in keys:
                if k not in ids:
                    raise AssertionError("node not found")
            total += len(keys)
        self.assertEqual(len(onx.graph.node), total)

    def test_split_big_model_small(self):
        onx = self.create_model(1)

        split = OnnxSplitting(onx)
        self.assertNotIn('R10', split.cutting_points)
        self.assertIn('R7', split.cutting_points)
        for i in range(len(split.segments)):
            ox, _ = split._make_onnx(i, i + 1)
            self.assertNotEmpty(ox.graph.input)
            self.assertNotEmpty(ox.graph.output)
            if i > 0:
                self.assertEqual(
                    split.segments[i].begin, ox.graph.input[0].name)
            if i < len(split.segments) - 1:
                self.assertEqual(
                    split.segments[i].end, ox.graph.output[0].name)

        parts = split_onnx(onx, 2, verbose=0)

        ids = set(n.SerializeToString() for n in onx.graph.node)
        total = 0
        for p in parts:
            self.assertIn('IADD', set(i.name for i in p.graph.initializer))
            keys = [n.SerializeToString() for n in p.graph.node]
            for k in keys:
                if k not in ids:
                    raise AssertionError("node not found")
            total += len(keys)
        self.assertEqual(len(onx.graph.node), total)

    def test_split_big_model_small_constant(self):
        onx = self.create_model(2)

        split = OnnxSplitting(onx)
        self.assertNotIn('R10', split.cutting_points)
        self.assertIn('R7', split.cutting_points)
        for i in range(len(split.segments)):
            ox, _ = split._make_onnx(i, i + 1)
            self.assertNotEmpty(ox.graph.input)
            self.assertNotEmpty(ox.graph.output)
            if i > 0:
                self.assertEqual(
                    split.segments[i].begin, ox.graph.input[0].name)
            if i < len(split.segments) - 1:
                self.assertEqual(
                    split.segments[i].end, ox.graph.output[0].name)

        parts = split_onnx(onx, 2, verbose=0)
        with open("dd.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        for i, p in enumerate(parts):
            with open(f"dd{i}.onnx", "wb") as f:
                f.write(p.SerializeToString())

        ids = set(n.SerializeToString() for n in onx.graph.node)
        total = 0
        for ip, p in enumerate(parts):
            if ip == 0:
                self.assertNotIn('IADD', set(
                    i.name for i in p.graph.initializer))
            co = set()
            for node in p.graph.node:
                if node.op_type == 'Constant':
                    co |= set(node.output)
            if ip == 1:
                self.assertIn("IADD", co)
            keys = [n.SerializeToString() for n in p.graph.node]
            for k in keys:
                if k not in ids:
                    raise AssertionError("node not found")
            total += len(keys)
        self.assertEqual(len(onx.graph.node), total - 1)

    def test_split_force(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None, None])
        T = make_tensor_value_info('T', TensorProto.FLOAT, [None, None])
        nodes = [make_node('Sub', ['X', 'Y'], ['diff']),
                 make_node('Mul', ['diff', 'diff'], ['abs']),
                 make_node('Neg', ['abs'], ['mabs']),
                 make_node('Add', ['mabs', 'Z'], ['dz1']),
                 make_node('Sub', ['mabs', 'Z'], ['dz2']),
                 make_node('Mul', ['dz1', 'dz2'], ['T'])]

        graph = make_graph(nodes, "dummy", [X, Y, Z], [T])
        onx = make_model(graph)
        check_model(onx)

        self.assertRaise(lambda: split_onnx(onx), ValueError)
        self.assertRaise(lambda: split_onnx(onx, []), TypeError)
        self.assertRaise(lambda: split_onnx(onx, cut_points=2), TypeError)
        self.assertRaise(lambda: split_onnx(onx, 2, []), ValueError)
        parts, stats = split_onnx(onx, 2, stats=True)
        self.assertEqual(len(parts), 2)
        self.assertEqual(stats["split_points"], ["mabs"])
        parts, stats = split_onnx(onx, cut_points=["abs"], stats=True)
        self.assertEqual(len(parts), 2)
        self.assertEqual(stats["split_points"], ["abs"])

    def test_split_input_optional(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Ic = make_tensor_value_info('I', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None, None])
        T = make_tensor_value_info('T', TensorProto.FLOAT, [None, None])
        nodes = [make_node('Sub', ['X', 'Y'], ['diff']),
                 make_node('Mul', ['diff', 'diff'], ['muld']),
                 make_node('Neg', ['muld'], ['oneg']),
                 make_node('Add', ['oneg', 'Z'], ['dz1']),
                 make_node('Sub', ['oneg', 'Z'], ['dz2']),
                 make_node('Add', ['dz1', 'I'], ['dz1I']),
                 make_node('Mul', ['dz1I', 'dz2'], ['T'])]
        Idef = numpy_helper.from_array(
            numpy.array([1], dtype=numpy.float32), name='I')

        graph = make_graph(nodes, "dummy", [X, Y, Z, Ic], [T], [Idef])
        onx = make_model(graph)
        check_model(onx)

        self.assertRaise(lambda: split_onnx(onx), ValueError)
        self.assertRaise(lambda: split_onnx(onx, []), TypeError)
        self.assertRaise(lambda: split_onnx(onx, cut_points=2), TypeError)
        self.assertRaise(lambda: split_onnx(onx, 2, []), ValueError)
        parts, stats = split_onnx(onx, 2, stats=True)
        self.assertEqual(len(parts), 2)
        for i, p in enumerate(parts):
            try:
                check_model(p)
            except Exception as e:
                with open(f"test_split_input_optional_{i}.onnx", "wb") as f:
                    f.write(p.SerializeToString())
                raise AssertionError(f"Part {i} is not valid.\n{p}") from e
        self.assertEqual(len(parts), 2)
        self.assertEqual(stats["split_points"], ["oneg"])
        names1 = [i.name for i in parts[0].graph.input]
        names2 = [i.name for i in parts[1].graph.input]
        self.assertEqual(["X", "Y"], names1)
        self.assertEqual(["oneg", "Z", "I"], names2)

    def test_split_input_optional_duplicated(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Ic = make_tensor_value_info('I', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None, None])
        T = make_tensor_value_info('T', TensorProto.FLOAT, [None, None])
        nodes = [make_node('Sub', ['X', 'Y'], ['diff']),
                 make_node('Mul', ['diff', 'diff'], ['muld']),
                 make_node('Add', ['muld', 'I'], ['muldi']),
                 make_node('Neg', ['muldi'], ['oneg']),
                 make_node('Add', ['oneg', 'Z'], ['dz1']),
                 make_node('Sub', ['oneg', 'Z'], ['dz2']),
                 make_node('Add', ['dz1', 'I'], ['dz1I']),
                 make_node('Mul', ['dz1I', 'dz2'], ['T'])]
        Idef = numpy_helper.from_array(
            numpy.array([1], dtype=numpy.float32), name='I')

        graph = make_graph(nodes, "dummy", [X, Y, Z, Ic], [T], [Idef])
        onx = make_model(graph)
        check_model(onx)

        self.assertRaise(lambda: split_onnx(onx), ValueError)
        self.assertRaise(lambda: split_onnx(onx, []), TypeError)
        self.assertRaise(lambda: split_onnx(onx, cut_points=2), TypeError)
        self.assertRaise(lambda: split_onnx(onx, 2, []), ValueError)
        parts, stats = split_onnx(onx, 2, stats=True)
        self.assertEqual(len(parts), 2)
        for i, p in enumerate(parts):
            try:
                check_model(p)
            except Exception as e:
                with open(f"test_split_input_optional_{i}.onnx", "wb") as f:
                    f.write(p.SerializeToString())
                raise AssertionError(f"Part {i} is not valid.\n{p}") from e
        self.assertEqual(len(parts), 2)
        self.assertEqual(stats["split_points"], ["oneg"])
        names1 = [i.name for i in parts[0].graph.input]
        names2 = [i.name for i in parts[1].graph.input]
        self.assertEqual(["X", "Y", "I"], names1)
        self.assertEqual(["oneg", "Z", "I"], names2)

    def test_split_input_duplicated(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Ic = make_tensor_value_info('I', TensorProto.FLOAT, [1])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None, None])
        T = make_tensor_value_info('T', TensorProto.FLOAT, [None, None])
        nodes = [make_node('Sub', ['X', 'Y'], ['diff']),
                 make_node('Mul', ['diff', 'diff'], ['muld']),
                 make_node('Add', ['muld', 'I'], ['muldi']),
                 make_node('Neg', ['muldi'], ['oneg']),
                 make_node('Add', ['oneg', 'Z'], ['dz1']),
                 make_node('Sub', ['oneg', 'Z'], ['dz2']),
                 make_node('Add', ['dz1', 'I'], ['dz1I']),
                 make_node('Mul', ['dz1I', 'dz2'], ['T'])]

        graph = make_graph(nodes, "dummy", [X, Y, Z, Ic], [T])
        onx = make_model(graph)
        check_model(onx)

        self.assertRaise(lambda: split_onnx(onx), ValueError)
        self.assertRaise(lambda: split_onnx(onx, []), TypeError)
        self.assertRaise(lambda: split_onnx(onx, cut_points=2), TypeError)
        self.assertRaise(lambda: split_onnx(onx, 2, []), ValueError)
        parts, stats = split_onnx(onx, stats=True, cut_points=["oneg"])
        self.assertEqual(len(parts), 2)
        for i, p in enumerate(parts):
            try:
                check_model(p)
            except Exception as e:
                with open(f"test_split_input_optional_{i}.onnx", "wb") as f:
                    f.write(p.SerializeToString())
                raise AssertionError(f"Part {i} is not valid.\n{p}") from e
        self.assertEqual(len(parts), 2)
        self.assertEqual(stats["split_points"], ["oneg"])
        names1 = [i.name for i in parts[0].graph.input]
        names2 = [i.name for i in parts[1].graph.input]
        self.assertEqual(["X", "Y", "I"], names1)
        self.assertEqual(["oneg", "Z", "I"], names2)

    def test_split_reshape(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None, None, 1])
        T = make_tensor_value_info('T', TensorProto.FLOAT, [None, None])
        nodes = [make_node('Sub', ['X', 'Y'], ['diff']),
                 make_node('Mul', ['diff', 'diff'], ['muld']),
                 make_node('Shape', ['muld'], ['shape']),
                 make_node('Concat', ['shape', 'I'], ['new_shape'], axis=0),
                 make_node('Reshape', ['muld', 'new_shape'], ['muldis']),
                 make_node('Neg', ['muldis'], ['oneg']),
                 make_node('Add', ['oneg', 'Z'], ['dz1']),
                 make_node('Sub', ['oneg', 'Z'], ['dz2']),
                 make_node('CastLike', ['I', 'dz1'], ['fI']),
                 make_node('Add', ['dz1', 'fI'], ['dz1I']),
                 make_node('Mul', ['dz1I', 'dz2'], ['tt']),
                 make_node('Reshape', ['tt', 'new_shape'], ['T'])]
        Idef = numpy_helper.from_array(
            numpy.array([1], dtype=numpy.int64), name='I')

        graph = make_graph(nodes, "dummy", [X, Y, Z], [T], [Idef])
        onx = make_model(graph, opset_imports=[make_opsetid('', 17)])
        check_model(onx)
        # with open("debug.onnx", "wb") as f:
        #     f.write(onx.SerializeToString())

        parts, stats = split_onnx(onx, 2, stats=True,
                                  doc_string=True, verbose=0)
        names = stats["shape_results"]
        self.assertEqual(names, {'I', 'new_shape', 'shape'})
        self.assertEqual(len(parts), 2)
        cuts = stats["cutting_points"]
        self.assertIn("oneg", cuts)

        for i, p in enumerate(parts):
            try:
                check_model(p)
            except Exception as e:
                with open(f"test_split_reshape_{i}.onnx", "wb") as f:
                    f.write(p.SerializeToString())
                raise AssertionError(f"Part {i} is not valid.\n{p}") from e
        self.assertEqual(len(parts), 2)
        self.assertEqual(stats["split_points"], ["oneg"])
        names1 = [i.name for i in parts[0].graph.input]
        names2 = [i.name for i in parts[1].graph.input]
        self.assertEqual(["X", "Y"], names1)
        self.assertEqual(["oneg", "Z", "new_shape"], names2)
        names1 = [i.name for i in parts[0].graph.output]
        names2 = [i.name for i in parts[1].graph.output]
        self.assertEqual(["T"], names2)
        self.assertEqual(["oneg", "new_shape"], names1)

        sess = InferenceSession(onx.SerializeToString(),
                                providers=['CPUExecutionProvider'])
        sesst = [InferenceSession(p.SerializeToString(),
                                  providers=['CPUExecutionProvider'])
                 for p in parts]
        x = numpy.arange(4).reshape((2, 2)).astype(numpy.float32)
        y = x + 1
        z = (x * 10).reshape((2, 2, 1))
        feeds = {'X': x, 'Y': y, 'Z': z}
        expected = sess.run(None, feeds)[0]
        del feeds['Z']
        oneg, new_shape = sesst[0].run(None, feeds)
        got = sesst[1].run(
            None, {'oneg': oneg, 'new_shape': new_shape, 'Z': z})
        self.assertEqualArray(expected, got)

    def test_split_reshape_back(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None, None, 1])
        T = make_tensor_value_info('T', TensorProto.FLOAT, [None, None])
        nodes = [make_node('Sub', ['X', 'Y'], ['diff']),
                 make_node('Mul', ['diff', 'diff'], ['muld']),
                 make_node('Shape', ['muld'], ['shape']),
                 make_node('Identity', ['Init'], ['Id']),
                 make_node('Concat', ['shape', 'Id'], ['new_shape'], axis=0),
                 make_node('Reshape', ['muld', 'new_shape'], ['muldis']),
                 make_node('Neg', ['muldis'], ['oneg']),
                 make_node('Add', ['oneg', 'Z'], ['dz1']),
                 make_node('Sub', ['oneg', 'Z'], ['dz2']),
                 make_node('CastLike', ['Init', 'dz1'], ['fI']),
                 make_node('Add', ['dz1', 'fI'], ['dz1I']),
                 make_node('Mul', ['dz1I', 'dz2'], ['tt']),
                 make_node('Reshape', ['tt', 'new_shape'], ['T'])]
        Idef = numpy_helper.from_array(
            numpy.array([1], dtype=numpy.int64), name='Init')

        graph = make_graph(nodes, "dummy", [X, Y, Z], [T], [Idef])
        onx = make_model(graph, opset_imports=[make_opsetid('', 17)])
        check_model(onx)
        # with open("debug.onnx", "wb") as f:
        #     f.write(onx.SerializeToString())

        parts, stats = split_onnx(onx, 2, stats=True,
                                  doc_string=True, verbose=0)
        self.assertEqual(['diff', 'muld', 'muldis', 'oneg', 'tt'],
                         stats["split"].cutting_points)
        msg = stats["split"].display_shape_node_result()
        seg = stats["split"].segments[2]
        self.assertEqual({'new_shape'},
                         seg.shape_results)
        self.assertIn("Shape|NS(muld) -> shape|S", msg)
        self.assertNotIn("Mul|NS(diff|S, diff|S) -> muld|S", msg)
        self.assertIn("Identity|NS(Init|S) -> Id|S", msg)
        self.assertIn("CastLike(Init|S, dz1) -> fI", msg)
        names = stats["shape_results"]
        self.assertEqual(names, {'new_shape', 'shape', 'Id', 'Init'})
        self.assertEqual(len(parts), 2)
        cuts = stats["cutting_points"]
        self.assertIn("oneg", cuts)

        for i, p in enumerate(parts):
            try:
                check_model(p)
            except Exception as e:
                with open("test_split_reshape_back.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
                with open(f"test_split_reshape_back_{i}.onnx", "wb") as f:
                    f.write(p.SerializeToString())
                raise AssertionError(f"Part {i} is not valid.\n{p}") from e
        self.assertEqual(len(parts), 2)
        self.assertEqual(stats["split_points"], ["oneg"])
        names1 = [i.name for i in parts[0].graph.input]
        names2 = [i.name for i in parts[1].graph.input]
        self.assertEqual(["X", "Y"], names1)
        self.assertEqual(["oneg", "Z", "new_shape"], names2)
        names1 = [i.name for i in parts[0].graph.output]
        names2 = [i.name for i in parts[1].graph.output]
        self.assertEqual(["T"], names2)
        self.assertEqual(["oneg", "new_shape"], names1)

        sess = InferenceSession(onx.SerializeToString(),
                                providers=['CPUExecutionProvider'])
        sesst = [InferenceSession(p.SerializeToString(),
                                  providers=['CPUExecutionProvider'])
                 for p in parts]
        x = numpy.arange(4).reshape((2, 2)).astype(numpy.float32)
        y = x + 1
        z = (x * 10).reshape((2, 2, 1))
        feeds = {'X': x, 'Y': y, 'Z': z}
        expected = sess.run(None, feeds)[0]
        del feeds['Z']
        oneg, new_shape = sesst[0].run(None, feeds)
        got = sesst[1].run(
            None, {'oneg': oneg, 'new_shape': new_shape, 'Z': z})
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    TestSplitOnnx().test_split_reshape_back()
    unittest.main(verbosity=2)
