"""
Create: 2022.05.24
Author: SG.SUH
Python: 3.8
PyTorch: 1.9
"""

import onnx_graphsurgeon as gs
import numpy as np

def loop_node(graph, 
              current_node, 
              loop_time = 0):
    for i in range(loop_time):
        next_node = [node for node in graph.nodes if len(node.inputs) != 0 and len(current_node.outputs) != 0 and node.inputs[0] == current_node.outputs[0]][0]
        current_node = next_node

    return next_node

def simplify_postprocess(onnx_model):
    print('Use onnx_graphsurgeon to adjust postprocessing part in the onnx...')

    graph = gs.import_onnx(onnx_model)

    cls_preds = gs.Variable(name = 'cls_preds', dtype = np.float32, shape = (1, 248, 216, 18))
    box_preds = gs.Variable(name = 'box_preds', dtype = np.float32, shape = (1, 248, 216, 42))
    dir_cls_preds = gs.Variable(name = 'dir_cls_preds', dtype = np.float32, shape = (1, 248, 216, 12))

    tmap = graph.tensors()
    new_inputs = [tmap['voxels'], tmap['voxel_idxs'], tmap['voxel_num']]
    new_outputs = [cls_preds, box_preds, dir_cls_preds]

    for inp in graph.inputs:
        if inp not in new_inputs:
            inp.outputs.clear()

    for out in graph.outputs:
        out.inputs.clear()

    first_ConvTranspose_node = [node for node in graph.nodes if node.op == 'ConvTranspose'][0]
    concat_node = loop_node(graph, first_ConvTranspose_node, 3)

    assert concat_node.op == 'Concat'

    first_node_after_concat = [node for node in graph.nodes if len(node.inputs) != 0 and len(concat_node.outputs) != 0 and node.inputs[0] == concat_node.outputs[0]]

    for i in range(3):
        transpose_node = loop_node(graph, first_node_after_concat[i], 1)

        assert transpose_node.op == 'Transpose'

        transpose_node.outputs = [new_outputs[i]]

    graph.inputs = new_inputs
    graph.outputs = new_outputs
    graph.cleanup().toposort()

    return gs.export_onnx(graph)

@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    for inp in inputs:
        inp.outputs.clear()

    for out in outputs:
        out.inputs.clear()

    op_attrs = dict()
    op_attrs['dense_shape'] = np.array([496, 432])

    return self.layer(name = 'PillarScatter_0', op = 'PillarScatterPlugin', inputs = inputs, outputs = outputs, attrs = op_attrs)

def simplify_preprocess(onnx_model):
    print('Use onnx_graphsurgeon to modify onnx...')

    graph = gs.import_onnx(onnx_model)

    tmap = graph.tensors()
    MAX_VOXELS = tmap['voxels'].shape[0]

    input_new = gs.Variable(name = 'voxels', dtype = np.float32, shape = (MAX_VOXELS, 32, 10))

    X = gs.Variable(name = 'voxel_idxs', dtype = np.int32, shape = (MAX_VOXELS, 4))

    Y = gs.Variable(name = 'voxel_num', dtype = np.int32, shape = (1,))

    first_node_after_pillarscatter = [node for node in graph.nodes if node.op == 'Conv'][0]

    first_node_pillarvfe = [node for node in graph.nodes if node.op == 'MatMul'][0]

    next_node = current_node = first_node_pillarvfe

    for i in range(6):
        next_node = [node for node in graph.nodes if node.inputs[0] == current_node.outputs[0]][0]

        if i == 5:
            current_node.attrs['keepdims'] = [0]
            
            break

        current_node = next_node

    last_node_pillarvfe = current_node

    graph.inputs.append(Y)

    inputs = [last_node_pillarvfe.outputs[0], X, Y]
    outputs = [first_node_after_pillarscatter.inputs[0]]

    graph.replace_with_clip(inputs, outputs)

    graph.cleanup().toposort()

    graph.inputs = [first_node_pillarvfe.inputs[0], X, Y]
    graph.outputs = [tmap['cls_preds'], tmap['box_preds'], tmap['dir_cls_preds']]

    graph.cleanup()

    graph.inputs = [input_new, X, Y]
    first_add = [node for node in graph.nodes if node.op == 'MatMul'][0]
    first_add.inputs[0] = input_new

    graph.cleanup().toposort()

    return gs.export_onnx(graph)