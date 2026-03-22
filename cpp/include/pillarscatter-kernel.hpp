/*
Create: 2026.03.21
Author: SG.SUH
Ref: NVIDIA-AI-IOT/CUDA-Pillars
*/

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

int pillarScatterHalfKernelLaunch(const half* pillar_features_data,
                                const unsigned int* coords_data,
                                const unsigned int* params_data,
                                unsigned int featureX,
                                unsigned int featureY,
                                half* spatial_feature_data,
                                cudaStream_t stream);

int pillarScatterFloatKernelLaunch(const float* pillar_features_data,
                                    const unsigned int* coords_data,
                                    const unsigned int* params_data,
                                    unsigned int featureX,
                                    unsigned int featureY,
                                    float* spatial_feature_data,
                                    cudaStream_t stream);