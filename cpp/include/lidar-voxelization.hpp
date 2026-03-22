/*
Create: 2026.03.21
Author: SG.SUH
Ref: NVIDIA-AI-IOT/CUDA-PointPillars
*/

#pragma once

#include <memory>
#include "dtype.hpp"

namespace pointpillar {
namespace lidar {
struct VoxelizationParameter {
    nvtype::Float3 min_range;
    nvtype::Float3 max_range;
    nvtype::Float3 voxel_size;
    nvtype::Int3 grid_size;
    int max_voxels;
    int max_points_per_voxel;
    int max_points;
    int num_feature;

    static nvtype::Int3 compute_grid_size(const nvtype::Float3& max_range, const nvtype::Float3& min_range, const nvtype::Float3& voxel_size);
};

class Voxelization {
public:
    virtual void forward(const float* points, int num_points, void* stream = nullptr) = 0;

    virtual const nvtype::half* features() = 0;
    virtual const unsigned int* coords() = 0;
    virtual const unsigned int* params() = 0;
};

std::shared_ptr<Voxelization> create_voxelization(VoxelizationParameter param);
};
};