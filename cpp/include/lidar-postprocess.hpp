/*
Create: 2026.03.02
Author: SG.SUH
Ref: NVIDIA-AI-IOT/CUDA-PointPillars
*/

#pragma once

#include <memory>
#include <vector>
#include "dtype.hpp"

namespace pointpillar {
namespace lidar {
struct BoundingBox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    BoundingBox() {

    }
    BoundingBox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_) 
        : x(x_)
        , y(y_)
        , z(z_)
        , w(w_)
        , l(l_)
        , h(h_)
        , rt(rt_)
        , id(id_)
        , score(score_) {

    }
};

struct PostProcessParameter {
    nvtype::Float3 min_range;
    nvtype::Float3 max_range;
    nvtype::Int2 feature_size;
    int num_classes = 3;
    int num_anchors = 6;
    int len_per_anchor = 4;
    float anchors[24] = {
        3.9, 1.6, 1.56, 0.0,
        3.9, 1.6, 1.56, 1.57,
        0.8, 0.6, 1.73, 0.0,
        0.8, 0.6, 1.73, 1.57,
        1.76, 0.6, 1.73, 0.0,
        1.76, 0.6, 1.73, 1.57
    };
    nvtype::Float3 anchor_bottom_heights{-1.78, -0.6, -0.6};
    int num_box_values = 7;
    float score_thresh = 0.1;
    float dir_offset = 0.78539;
    float nms_thresh = 0.01;
};

class PostProcess {
public:
    virtual void forward(const float* cls, const float* box, const float* dir, void* stream) = 0;

    virtual std::vector<BoundingBox> bndBoxVec() = 0;
};

std::shared_ptr<PostProcess> create_postprocess(const PostProcessParameter& param);
};
};