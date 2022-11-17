"""
@file
@brief Helpers to split an ONNX models.
"""
import textwrap
import numpy
from onnx import ModelProto, shape_inference, TensorProto
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
        if begin is None and end is None:
            raise ValueError(f"A segment cannot contain this whole model.")
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
        return f"{node.op_type}-{node.name}-{idn}"

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

        # only working for standard domain (supporting shape inference)
        for node in onnx_model.graph.node:
            if node.domain not in {'', 'ai.onnx', 'ai.onnx.ml'}:
                raise NotImplementedError(
                    f"Node {node.op_type!r} from domain {node.domain!r} "
                    f"is not supported yet.")

        # cut points: results breaking the connexity of the graph
        self.cutting_points = self._get_cutting_points(node_list)

        if self.verbose:
            self.fLOG(
                f"[OnnxSplitting] # cuttings points: {len(self.cutting_points)}")

        # segments
        segments = []
        for i in range(len(self.cutting_points)):  # pylint: disable=C0200
            segments.append(
                self._make_segment(
                    None if i == 0 else self.cutting_points[i - 1],
                    self.cutting_points[i]))
        segments.append(self._make_segment(self.cutting_points[i], None))
        self.segments = segments
        self.shapes = shape_inference.infer_shapes(onnx_model)

        if self.verbose > 0:
            sizes = [seg.size for seg in self.segments]
            self.fLOG(f"[OnnxSplitting] # segments = {len(sizes)}, "
                      f"min,avg,max size=[{min(sizes)}, "
                      f"{sum(sizes) / len(sizes)}, {max(sizes)}]")

    @staticmethod
    def _connex_components(vertices, adja):
        vert = {v: i for i, v in enumerate(vertices)}
        more = True
        while more:
            more = False
            for k, v in adja.items():
                if v == 0:
                    continue
                a, b = k
                if vert[a] == vert[b]:
                    continue
                more = True
                if vert[a] < vert[b]:
                    vert[b] = vert[a]
                else:
                    vert[a] = vert[b]
        return vert

    @staticmethod
    def is_small(tensor):
        """
        Tells if a tensor is small. In that case, all edges to this
        constant are ignored when looking for cutting points.
        The algorithm assumes it can be duplicated in multiple parts.
        It is usually single float constant or shapes.
        """
        if tensor.HasField("segment"):
            raise ValueError("Currently not supporting loading segments.")
        if tensor.data_type == TensorProto.UNDEFINED:  # pylint: disable=E1101
            raise TypeError(
                "The element type in the input tensor is not defined.")

        dims = tensor.dims
        total = numpy.prod(dims)
        if total < 32:
            # Covers small constants, reshaping...
            return True
        return False

    def _get_cutting_points(self, node_list):
        # let's avoid adding small constant
        inits = {i.name: self.is_small(i)
                 for i in self.onnx_model.graph.initializer}
        inits.update({i.name: self.is_small(i)
                     for i in self.onnx_model.graph.sparse_initializer})
        set_small = set(k for k, v in inits.items() if v)
        for idn, node in node_list:
            if len(node.input) == 0 and len(node.SerializeToString()) < 128:
                key = self._key(idn, node)
                set_small.add(key)
                set_small |= set(node.output)

        # adjacency matrix
        constant_type = {'Constant', 'ConstantOfShape'}
        adja = {}
        vertices = set()
        ordered_names = []
        for idn, node in node_list:
            key = self._key(idn, node)
            if key in set_small:
                continue
            if (node.op_type not in constant_type and
                    len(node.output) == 1 and
                    len(node.input) > 0):
                # only single output can be cutting points
                ordered_names.extend(node.output)
            vertices.add(key)
            vertices |= set(i for i in node.input if i not in set_small)
            vertices |= set(o for o in node.output if o not in set_small)
            for i in node.input:
                if i in set_small:
                    continue
                adja[i, key] = 1
            for o in node.output:
                if o in set_small:
                    continue
                adja[key, o] = 1

        # checking the connexity
        cutting_points = []
        for name in ordered_names:
            keys = []
            for a, b in adja:
                if b == name:
                    keys.append((a, b))

            # remove the links
            for a, b in keys:
                adja[a, b] = 0

            connex = self._connex_components(vertices, adja)
            connex_id = set(connex.values())
            if len(connex_id) == 2:
                cutting_points.append(name)

            # put back the links
            for a, b in keys:
                adja[a, b] = 1

        return cutting_points

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
            if self.verbose > 0:
                n_nodes = len(self.onnx_model.graph.node)
                total = sum(s.size for s in self.segments)
                size = sum(self.segments[i].size for i in range(a, b))
                self.fLOG(f"[OnnxSplitting] part {i}: "
                          f"#nodes={len(onx.graph.node)}"  # pylint: disable=E1101
                          f"/{n_nodes}, size={size}/{total}={size/total:1.2f}")
        return res

    def _make_onnx(self, a, b, index=None):
        """
        Builds one onnx subpart including segments from a to b (excluded).
        """
        if index is None:
            index = a

        # common parts
        value_info = {o.name: o for o in self.onnx_model.graph.output}
        value_info.update({
            info.name: info
            for info in self.shapes.graph.value_info})  # pylint: disable=E1101

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
    if verbose > 0:
        (fLOG or print)(
                f"[split_onnx] starts splitting "
                f"{len(onnx_model.graph.node)} nodes in {n_parts} parts.")
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
