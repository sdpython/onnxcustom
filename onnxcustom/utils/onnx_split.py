"""
@file
@brief Helpers to split an ONNX models.
"""
from onnx import ModelProto


class OnnxSegment:
    """
    A segments of an onnx graph assuming
    it is the concatenation of all segments.
    
    :param parent: an instance of OnnxSplitting
    :param begin: names of the first extremity,
        None for the inputs of the main graph
    :param end: names of the second extremity,
        None for the outputs of the main graph
    :param inputs: needed inputs (if begin is None)
    :param inits: needed initializers
    :param sparse_inits: needed sparse initializers
    :param outputs: produced outputs (if end is None)
    """
    
    def __init__(self, parent, begin, end, 
                 inputs=None, inits=None, sparse_inits=None, outputs=None):
        if ((begin is None and inputs is None) or
                (begin is not None and inputs is not None)):
            raise ValueError("Only one among inputs or begin must be None.")
        if ((end is None and outputs is None) or
                (end is not None and outputs is not None)):
            raise ValueError("Only one among outputs or end must be None.")
        self.parent = parent
        self.begin = begin if isinstance(begin, list) else [begin]
        self.end = end if isinstance(end, list) else [end]
        self.inputs = inputs
        self.outputs = outputs
        self.inits = inits
        self.sparse_inits = inits


class OnnxSplitting:
    """
    The final goal is to split an onnx model into
    equivalent pieces.
    
    :param onnx_model: onnx_model    
    """
    
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model
        self._init()
        
    def _init(self):
        onnx_model = self.onnx_model
        if not isinstance(onnx_model, ModelProto):
            raise TypeError(f"onnx_model must a ModelProto not a {type(model)}.")
        node_list = [(i, node) for i, node in enumerate(onnx_model.graph.node)]

        # sizes
        sizes = {}
        for init in onnx_model.graph.initializer:
            sizes[init.name] = len(init.SerializeToString())
        for init in onnx_model.graph.sparse_initializer:
            sizes[init.name] = len(init.SerializeToString())
        
        for idn, node in node_list:
            sizes[f"{node.name}-{idn}"] = len(node.SerializeToString())
        self.sizes = sizes

        # output needs once
        consumed = {}
        for node in onnx_model.graph.node:
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
        self.cutting_points = cutting_points

        # segments
        segments = []
        for i in range(len(cutting_points)):
            segments.append(
                self._make_segment(
                    None if i == 0 else cutting_points[i-1],
                    cutting_points[i]))
        segments.append(self._make_segment(cutting_points[i], None))
        self.segments = segments
        
    def _make_segment(self, name1, name2):
        begin = name1
        end = name2
        if name1 is None:
            assert name2 is not None
            inputs = [i.name for i in self.onnx_model.graph.input]
            outputs = None
            needed = set()
            found = None
            nodes = []
            for idn, node in enumerate(self.onnx_model.graph.node):
                nodes.append((idn,node))
                if name2 in node.output:
                    found = idn, node
                    break
                    
            assert found is not None
            assert len(found[1].output) == 1
            assert found[1].output[0] == name2
            names = {name2}
            for idn, node in reversed(nodes):
                if set(node.output) & names:
                    names |= set(node.input)
            self.involded = names
            
        elif name2 is None:
            assert name1 is not None
            inputs = None
            outputs = [i.name for i in self.onnx_model.graph.output]
            raise NotImplementedError()
            
        else:
            assert name1 is not None
            assert name2 is not None
            inputs = None
            outputs = None
            raise NotImplementedError()
        


def split_onnx(onnx_model, n_parts):
    """
    Splits an ONNX model into *n_parts* consecutive subgraphs.
    Chained altogether, they are equivalent to the given model.

    :param onnx_model: onnx model
    :param n_parts: number of subgraphs
    :return: list of onnx model
    """
    spl_onnx = OnnxSplitting(onnx_model)
    if len(spl_onnx.cutting_points) < n_parts:
        raise RuntimeError(  # pragma: no cover
            f"Unable to split the onnn model, there are less cutting points "
            f"{len(spl_onnx.cutting_points)} than the number of requested "
            f"splits ({n_parts}).")
    return spl_onnx
