"""
Create: 2022.05.23
Author: SG.SUH
Python: 3.8
PyTorch: 1.9
"""

import argparse
import glob
import numpy as np
import torch
import onnx

from pathlib import Path
from onnxsim import simplify

from pcdet.config import cfg_from_yaml_file
from pcdet.config import cfg
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.utils.common_utils import create_logger
from pcdet.models import build_network

from simplifier_onnx import simplify_postprocess
from simplifier_onnx import simplify_preprocess


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_file', 
                        type = str, 
                        default = 'tools/cfgs/kitti_models/pointpillar.yaml')
    parser.add_argument('--data_path', 
                        type = str, 
                        default = 'kitti/3d_object/testing/velodyne')
    parser.add_argument('--ckpt', 
                        type = str, 
                        default = 'weights/pointpillar_7728.pth')
    parser.add_argument('--ext', 
                        type = str, 
                        default = '.bin')
    parser.add_argument('--num_classes', 
                        type = int, 
                        default = 3)
    parser.add_argument('--class_name', 
                        type = list, 
                        default = ['Car', 'Pedestrian', 'Cyclist'])
    parser.add_argument('--min_x_range', 
                        type = float, 
                        default = 0.0)
    parser.add_argument('--max_x_range', 
                        type = float, 
                        default = 69.12)
    parser.add_argument('--min_y_range', 
                        type = float, 
                        default = -39.68)
    parser.add_argument('--max_y_range', 
                        type = float, 
                        default = 39.68)
    parser.add_argument('--min_z_range', 
                        type = float, 
                        default = -3.0)
    parser.add_argument('--max_z_range', 
                        type = float, 
                        default = 1.0)
    parser.add_argument('--pillar_x_size', 
                        type = float, 
                        default = 0.16)
    parser.add_argument('--pillar_y_size', 
                        type = float, 
                        default = 0.16)
    parser.add_argument('--pillar_z_size', 
                        type = float, 
                        default = 4.0)
    parser.add_argument('--max_num_points_per_pillar', 
                        type = int, 
                        default = 32)
    parser.add_argument('--num_point_values', 
                        type = int, 
                        default = 4)
    parser.add_argument('--num_feature_scatter', 
                        type = int, 
                        default = 64)
    parser.add_argument('--dir_offset', 
                        type = float, 
                        default = 0.78539)
    parser.add_argument('--dir_limit_offset', 
                        type = float, 
                        default = 0.0)
    parser.add_argument('--num_dir_bins', 
                        type = int, 
                        default = 2)
    parser.add_argument('--anchor_sizes', 
                        type = list, 
                        default = [[[3.9, 1.6, 1.56]], [[0.8, 0.6, 1.73]], [[1.76, 0.6, 1.73]]])
    parser.add_argument('--anchor_rotations', 
                        type = list, 
                        default = [[0, 1.57], [0, 1.57], [0, 1.57]])
    parser.add_argument('--anchor_bottom_heights', 
                        type = list, 
                        default = [[-1.78], [-0.6], [-0.6]])
    parser.add_argument('--score_thresh', 
                        type = float, 
                        default = 0.1)
    parser.add_argument('--nms_thresh', 
                        type = float, 
                        default = 0.01)
    parser.add_argument('--train_max_voxels', 
                        type = int, 
                        default = 16000)
    parser.add_argument('--test_max_voxels', 
                        type = int, 
                        default = 40000)
    parser.add_argument('--export_onnx', 
                        type = bool, 
                        default = True)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

class DemoDataset(DatasetTemplate):
    def __init__(self, 
                 args, 
                 class_names, 
                 training = True, 
                 root_path = None, 
                 logger = None, 
                 ext = '.bin'):
        super().__init__(args, 
                         class_names = class_names, 
                         training = training, 
                         root_path = root_path, 
                         logger = logger)

        self.args = args
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / '*{}'.format(self.ext))) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()

        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype = np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {'points': points, 'framd_id': index}

        data_dict = self.prepare_data(data_dict = input_dict)

        return data_dict

def main():
    args, cfg = parse_config()

    logger = create_logger()

    logger.info('------ Convert OpenPCDet model for TensorRT ------')

    demo_dataset = DemoDataset(args, class_names = args.class_name, training = False, root_path = Path(args.data_path), ext = args.ext, logger = logger)

    model = build_network(args, num_class = args.num_classes, dataset = demo_dataset)

    model.load_params_from_file(filename = args.ckpt, logger = logger, to_cpu = True)
    model.cuda()
    model.eval()

    np.set_printoptions(threshold=np.inf)

    with torch.no_grad():
        MAX_VOXELS = 40000

        dummy_voxels = torch.zeros((MAX_VOXELS, 32, 4), dtype = torch.float32, device = 'cuda:0')
        dummy_voxel_idxs = torch.zeros((MAX_VOXELS, 4), dtype = torch.int32, device= 'cuda:0')
        dummy_voxel_num = torch.zeros((MAX_VOXELS), dtype = torch.int32, device = 'cuda:0')

        torch.onnx.export(model,       # model being run
          (dummy_voxels, dummy_voxel_num, dummy_voxel_idxs),               # model input (or a tuple for multiple inputs)
          "./pointpillar_raw.onnx",  # where to save the model (can be a file or file-like object)
          export_params = True,        # store the trained parameter weights inside the model file
          opset_version = 11,          # the ONNX version to export the model to
          do_constant_folding = True,  # whether to execute constant folding for optimization
          keep_initializers_as_inputs = True,
          verbose = True,
          input_names = ['voxels', 'voxel_num', 'voxel_idxs'],   # the model's input names
          output_names = ['cls_preds', 'box_preds', 'dir_cls_preds'], # the model's output names
        )

        onnx_raw = onnx.load('./pointpillar_raw.onnx')  # load onnx model
        onnx_trim_post = simplify_postprocess(onnx_raw)

        onnx_simp, check = simplify(onnx_trim_post)

        assert check, 'Simplified ONNX model could not be validated'

        onnx_final = simplify_preprocess(onnx_simp)

        onnx.save(onnx_final, 'pointpillar.onnx')

        print('finished exporting onnx')

    logger.info('[PASS] ONNX EXPORTED.')

if __name__ == '__main__':
    main()