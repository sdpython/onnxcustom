"""
@file
@brief Helpers to split an ONNX models.
"""
import textwrap
import numpy
from onnx import ModelProto, shape_inference
from onnx.helper import make_graph, make_model


class OnnxSegment:
    """
    A segments of an onnx graph assuming
    it is the concatenation of all segments.

    :param parent: an instance of OnnxSplitting
    :param begin: names of the first extremity,
        None for the inputs of the main graph
    :param end: names of the second extremity,
        None for the outputs of the main graph
    :param size: total size of the segment
    :param involved: list of result names involved in this segment
    :param nodes: involved nodes, list of tuple `(int, NodeProt)`
    """

    def __init__(self, parent, begin, end, size=0, involved=None, nodes=None):
        if begin is not None and not isinstance(begin, str):
            raise ValueError(f"begin={begin!r} must be a string or None.")
        if end is not None and not isinstance(end, str):
            raise ValueError(f"end={end!r} must be a string or None.")
        self.parent = parent
        self.begin = begin
        self.end = end
        self.involved = involved
        self.size = size
        self.nodes = nodes

    def __repr__(self):
        return f"{self.__class__.__name__}(...,\n    " + "\n".join(
            textwrap.wrap(
                f"{self.begin!r}, {self.end!r}, size={self.size!r}, "
                f"{self.involved!r})", subsequent_indent="    "))


class OnnxSplitting:
    """
    The final goal is to split an onnx model into
    equivalent pieces.

    :param onnx_model: onnx_model
    :param verbose: displays information during the split
    :param fLOG: logging function
    """

    def __init__(self, onnx_model, verbose=0, fLOG=None):
        self.onnx_model = onnx_model
        self.verbose = verbose
        self.fLOG = fLOG or print
        self._init()

    @staticmethod
    def _key(idn, node):
        return f"{node.name}-{idn}"

    def _init(self):
        onnx_model = self.onnx_model
        if not isinstance(onnx_model, ModelProto):
            raise TypeError(
                f"onnx_model must a ModelProto not a {type(onnx_model)}.")
        node_list = list(enumerate(onnx_model.graph.node))

        # sizes
        sizes = {}
        for init in onnx_model.graph.initializer:
            sizes[init.name] = len(init.SerializeToString())
        for init in onnx_model.graph.sparse_initializer:
            sizes[init.name] = len(init.SerializeToString())

        for idn, node in node_list:
            sizes[self._key(idn, node)] = len(node.SerializeToString())
        self.sizes = sizes

        # output needs once
        consumed = {}
        for node in onnx_model.graph.node:
            if node.domain not in {'', 'ai.onnx', 'ai.onnx.ml'}:
                raise NotImplementedError(
                    f"Node {node.op_type!r} from domain {node.domain!r} "
                    f"is not supported yet.")
            for i in node.input:
                if i not in consumed:
                    consumed[i] = 0
                consumed[i] += 1
            for o in node.output:
                consumed[o] = 0

        # cut points: unique output consumed by only one node
        cutting_points = []
        for idn, node in node_list:
            if len(node.output) != 1:
                continue
            out = node.output[0]
            if consumed[out] == 1:
                cutting_points.append(out)
        import pprint
        pprint.pprint(consumed)
        self.cutting_points = cutting_points

        if self.verbose:
            self.fLOG(f"[OnnxSplitting] # cuttings points: {len(self.cutting_points)}")

        # segments
        segments = []
        for i in range(len(cutting_points)):  # pylint: disable=C0200
            segments.append(
                self._make_segment(
                    None if i == 0 else cutting_points[i - 1],
                    cutting_points[i]))
        segments.append(self._make_segment(cutting_points[i], None))
        self.segments = segments
        self.shapes = shape_inference.infer_shapes(onnx_model)

        if self.verbose > 0:
            sizes = [seg.size for seg in self.segments]
            self.fLOG(f"[OnnxSplitting] # segments = {len(sizes)}, "
                      f"min,avg,max size=[{min(sizes)}, {sum(sizes) / len(sizes)}, {max(sizes)}]")

    def _make_segment(self, name1, name2):
        nodes = []
        for idn, node in enumerate(self.onnx_model.graph.node):
            nodes.append((idn, node))
            if name2 is not None and name2 in node.output:
                break

        if name2 is None:
            names = set(i.name for i in self.onnx_model.graph.output)
        else:
            names = {name2}

        size = 0
        subset = []
        for idn, node in reversed(nodes):
            if set(node.output) & names:
                size += self.sizes[self._key(idn, node)]
                if len(node.output) == 1 and node.output[0] == name1:
                    continue
                subset.append((idn, node))
                if len(node.input) == 1 and node.input[0] == name1:
                    continue
                for i in node.input:
                    if i in self.sizes:
                        size += self.sizes[i]
                names |= set(node.input)
        involved = names if name2 is None else names - {name2}
        return OnnxSegment(self, begin=name1, end=name2, involved=involved,
                           size=size, nodes=subset)

    def _split_2(self, a, b):
        """
        Splits the segments into two groups of the same size.

        :param a: first segment (included)
        :param b: second segment (excluded)
        :return: split index
        """
        if a >= b - 1:
            raise RuntimeError(f"a={a}, b={b}, unable to split.")
        if a == b - 2:
            return a + 1
        sizes = numpy.array([s.size for s in self.segments[a:b]])
        sizes_for = numpy.cumsum(sizes)
        sizes_bck = numpy.cumsum(sizes[::-1])[::-1]
        diff = numpy.abs(sizes_bck - sizes_for)
        pos = numpy.argmin(diff)
        pos += a
        if pos == a:
            pos = a + 1
        elif pos == b:
            pos = b - 1
        return pos

    def split_segment(self, n_parts):
        """
        Splits the segments into `n_parts` segments

        :param n_parts: number of parts to get
        :return: list of segments indices
        """
        extremities = [0, len(self.segments)]
        n = n_parts
        while n > 1:
            if n % 2 != 0:
                raise NotImplementedError(
                    f"n_parts={n_parts} is not a power of 2.")
            new_ext = [extremities[0]]
            for i in range(1, len(extremities)):
                a, b = extremities[i - 1:i + 1]
                pos = self._split_2(a, b)
                new_ext.extend([pos, b])
            extremities = new_ext
            n = n // 2
        return extremities

    def make_onnx(self, extremities):
        """
        Builds onnx subparts based on the segmentation
        defined by extremities.

        :param extremities: example, `[0, 3, 5]`,
            first onnx part contains segments `0:3=[0, 1, 2]`,
            second onnx part contains segments `3:5=[3, 4]`
        :return: list of onnx subgraphs (:epkg:`ModelProto`)
        """
        res = []
        for i in range(1, len(extremities)):
            a, b = extremities[i - 1:i + 1]
            onx = self._make_onnx(a, b, i - 1)
            res.append(onx)
        return res

    def _make_onnx(self, a, b, index):
        """
        Builds one onnx subpart including segments from a to b (excluded).
        """
        # common parts
        value_info = {info.name: info
                      for info in self.shapes.graph.value_info}  # pylint: disable=E1101

        segs = self.segments[a:b]
        involved = set()
        for seg in segs:
            involved |= seg.involved

        # initiliazers
        new_inits = [init for init in self.onnx_model.graph.initializer
                     if init.name in involved]
        new_sp_inits = [init for init in self.onnx_model.graph.sparse_initializer
                        if init.name in involved]

        # nodes
        nodes = []
        for seg in segs:
            for _, node in seg.nodes:
                nodes.append(node)

        # inputs, outputs
        if a == 0:
            new_inputs = [
                i for i in self.onnx_model.graph.input if i.name in involved]
        else:
            new_inputs = [value_info[segs[0].begin]]

        if b == len(self.segments):
            new_outputs = [
                i for i in self.onnx_model.graph.output if i.name in involved]
        else:
            new_outputs = [value_info[segs[-1].end]]

        model = self.onnx_model
        graph = make_graph(
            nodes, f"{model.graph.name}-{index}",
            new_inputs, new_outputs,
            new_inits, doc_string=model.graph.doc_string,
            sparse_initializer=new_sp_inits,
            value_info=model.graph.value_info)
        new_model = make_model(graph, opset_imports=model.opset_import)
        new_model.ir_version = model.ir_version
        new_model.producer_name = model.producer_name
        new_model.producer_version = model.producer_version
        new_model.domain = model.domain
        new_model.model_version = model.model_version
        new_model.doc_string = model.doc_string
        return new_model


def split_onnx(onnx_model, n_parts, verbose=0, fLOG=None):
    """
    Splits an ONNX model into *n_parts* consecutive subgraphs.
    Chained altogether, they are equivalent to the given model.

    :param onnx_model: onnx model
    :param n_parts: number of subgraphs
    :param verbose: display information related to the split
    :param fLOG: logging function
    :return: list of onnx model
    """
    if len(onnx_model.functions) > 0:
        raise NotImplementedError(
            f"The function does not work if the model contains function: "
            f"{f.name for f in onnx_model.functions}.")
    spl_onnx = OnnxSplitting(onnx_model, verbose=verbose, fLOG=fLOG or print)
    if len(spl_onnx.cutting_points) < n_parts:
        raise RuntimeError(  # pragma: no cover
            f"Unable to split the onnn model, there are less cutting points "
            f"{len(spl_onnx.cutting_points)} than the number of requested "
            f"splits ({n_parts}).")
    exts = spl_onnx.split_segment(n_parts)
    if verbose > 0:
        names = [spl_onnx.segments[i].end for i in exts[1:-1]]
        (fLOG or print)(f"[split_onnx] splits: {exts}, names={names}")
    return spl_onnx.make_onnx(exts)
