/*
Create: 2026.03.21
Author: SG.SUH
Ref: NVIDIA-AI-IOT/CUDA-PointPillars
*/

#pragma once

#include "lidar-voxelization.hpp"
#include "lidar-backbone.hpp"
#include "lidar-postprocess.hpp"

namespace pointpillar {
namespace lidar {
struct CoreParameter {
    VoxelizationParameter voxelization;
    std::string lidar_model;
    PostProcessParameter lidar_post;
};

class Core {
public:
    virtual std::vector<BoundingBox> forward(const float* lidar_points, int num_points, void* stream) = 0;

    virtual void print() = 0;
    virtual void set_timer(bool enable) = 0;
};

std::shared_ptr<Core> create_core(const CoreParameter& param);
};
};