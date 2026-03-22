/*
Create: 2026.03.02
Author: SG.SUH
Ref: NVIDIA-AI-IOT/CUDA-PointPillars
*/

#include <cuda_fp16.h>

#include <numeric>

#include "lidar-backbone.hpp"
#include "check.hpp"
#include "launch.cuh"
#include "tensorrt.hpp"

namespace pointpillar {
namespace lidar {
class BackboneImplement : public Backbone {
public:
    virtual ~BackboneImplement() {
        if(cls_) {
            checkRuntime(cudaFree(cls_));
        }
        if(box_) {
            checkRuntime(cudaFree(box_));
        }
        if(dir_) {
            checkRuntime(cudaFree(dir_));
        }
    }

    bool init(const std::string& model) {
        engine_ = TensorRT::load(model);
        if(engine_ == nullptr) {
            return false;
        }

        cls_dims_ = engine_->static_dims(3);
        box_dims_ = engine_->static_dims(4);
        dir_dims_ = engine_->static_dims(5);

        int32_t volumn = std::accumulate(cls_dims_.begin(), cls_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&cls_, volumn * sizeof(float)));

        volumn = std::accumulate(box_dims_.begin(), box_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&box_, volumn * sizeof(float)));

        volumn = std::accumulate(dir_dims_.begin(), dir_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&dir_, volumn * sizeof(float)));
        return true;
    }

    virtual void print() override {
        engine_->print("Lidar Backbone");
    }

    virtual void forward(const nvtype::half* voxels, const unsigned int* voxel_idxs, const unsigned int* params, void* stream = nullptr) override {
        cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
        engine_->forward({voxels, voxel_idxs, params, cls_, box_, dir_}, static_cast<cudaStream_t>(_stream));
    }

    virtual float* cls() override {
        return cls_;
    }
    virtual float* box() override {
        return box_;
    }
    virtual float* dir() override {
        return dir_;
    }

private:
    std::shared_ptr<TensorRT::Engine> engine_;
    float* cls_ = nullptr;
    float* box_ = nullptr;
    float* dir_ = nullptr;
    std::vector<int> cls_dims_;
    std::vector<int> box_dims_;
    std::vector<int> dir_dims_;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model) {
    std::shared_ptr<BackboneImplement> instance(new BackboneImplement());
    if(!instance->init(model)) {
        instance.reset();
    }
    return instance;
}
};
};