"""
Create: 2026.02.19
Author: SG.SUH
Python: 3.10
Ref: NVIDIA-AI-IOT/CUDA-PointPillars
"""

import os
import glob 
import onnx 
import torch 
import argparse
import numpy as np

from pathlib import Path
from onnxsim import simplify
from pcdet.utils import common_utils
from pcdet.models import build_network
from pcdet.datasets import DatasetTemplate
from pcdet.config import (
    cfg,
    cfg_from_yaml_file
)

from modify_onnx import (
    simplify_preprocess,
    simplify_postprocess
)

import sys

sys.path.append(".")


class PointPillarONNXWrapper(torch.nn.Module):
    """Wrapper to export PointPillar to ONNX.

    torch.onnx.export uses tracing which flattens dict inputs into individual
    tensors, breaking forward(batch_dict). This wrapper accepts individual
    tensors, builds batch_dict internally, runs the model modules, and returns
    raw prediction tensors (no post-processing) compatible with ONNX tracing.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxels, voxel_num_points, voxel_coords):
        batch_dict = {
            'voxels': voxels,
            'voxel_num_points': voxel_num_points,
            'voxel_coords': voxel_coords,
            'batch_size': 1,
        }
        for module in self.model.module_list:
            batch_dict = module(batch_dict)

        ret = self.model.dense_head.forward_ret_dict
        return ret['cls_preds'], ret['box_preds'], ret['dir_cls_preds']

class DemoDataset(DatasetTemplate):
    def __init__(self,
                 dataset_cfg,
                 class_names,
                 training=True,
                 root_path=None,
                 logger=None,
                 ext=".bin"):
        super().__init__(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         training=training,
                         root_path=root_path,
                         logger=logger)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)
    
    def __getitem__(self,
                    index):
        if self.ext == ".bin":
            points = np.fromfile(self.sample_file_list[index],
                                 dtype=np.float32).reshape(-1, 4)
        elif self.ext == ".npy":
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplemented
        
        input_dict = {"points": points,
                      "frame_id": index}
        
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--cfg_file",
                        type=str,
                        default="tools/cfgs/kitti_models/pointpillar.yaml",
                        help="specify the config for demo")
    parser.add_argument("--data_path",
                        type=str,
                        default="cpp/data",
                        help="specify the point cloud data file or directory")
    parser.add_argument("--ckpt",
                        type=str,
                        default="tools/weights/pointpillar_7728.pth",
                        help="specify the pretrained model")
    parser.add_argument("--out_dir",
                        type=str,
                        default="cpp/model",
                        help="specify the extension of your point cloud data file")
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file,
                       cfg)
    
    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("------ Convert OpenPCDet model for TensorRT ------")
    demo_dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG,
                               class_names=cfg.CLASS_NAMES,
                               training=False,
                               root_path=Path(args.data_path),
                               ext=".bin",
                               logger=logger)
    
    model = build_network(model_cfg=cfg.MODEL,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt,
                                logger=logger,
                                to_cpu=True)
    
    model.cuda()
    model.eval()

    wrapper = PointPillarONNXWrapper(model)
    wrapper.cuda()
    wrapper.eval()

    np.set_printoptions(threshold=np.inf)

    with torch.no_grad():
        MAX_VOXELS = 10000

        dummy_voxels = torch.zeros((MAX_VOXELS, 32, 4),
                                   dtype=torch.float32,
                                   device="cuda:0")

        dummy_voxel_idxs = torch.zeros((MAX_VOXELS, 4),
                                       dtype=torch.int32,
                                       device="cuda:0")

        dummy_voxel_num = torch.zeros((MAX_VOXELS,),
                                      dtype=torch.int32,
                                      device="cuda:0")

        torch.onnx.export(wrapper,
                          (dummy_voxels, dummy_voxel_num, dummy_voxel_idxs),
                          os.path.join(args.out_dir,
                                       "pointpillar_raw.onnx"),
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          keep_initializers_as_inputs=True,
                          input_names=["voxels", "voxel_num", "voxel_idxs"],
                          output_names=["cls_preds", "box_preds", "dir_cls_preds"])
        
        onnx_raw = onnx.load(os.path.join(args.out_dir, "pointpillar_raw.onnx"))
        onnx_trim_post = simplify_postprocess(onnx_raw)

        onnx_simp, check = simplify(onnx_trim_post)
        assert check, "Simplified ONNX model could not be validated"

        onnx_final = simplify_preprocess(onnx_simp)
        onnx.save(onnx_final,
                  os.path.join(args.out_dir, "pointpillar.onnx"))
        
        logger.info("[PASS] ONNX EXPORTED.")

if __name__ == "__main__":
    main()