/*
Create: 2026.03.02
Author: SG.SUH
Ref: NVIDIA-AI-IOT/CUDA-PointPillars
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dtype.hpp"

namespace pointpillar {
namespace lidar {
class Backbone {
public:
    virtual void forward(const nvtype::half* voxels, const unsigned int* voxel_idxs, const unsigned int* params, void* stream = nullptr) = 0;

    virtual float* cls() = 0;
    virtual float* box() = 0;
    virtual float* dir() = 0;

    virtual void print() = 0;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model);
};
};