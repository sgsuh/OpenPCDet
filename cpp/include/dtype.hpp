/*
Create: 2026.02.20
Author: SG.SUH
Ref: NVIDIA-AI-IOT/CUDA-PointPillars
*/

#pragma once

namespace nvtype {
struct Int2 {
    int x;
    int y;

    Int2() = default;
    Int2(int x, int y = 0) 
        : x(x)
        , y(y) {

    }
};

struct Int3 {
    int x;
    int y;
    int z;

    Int3() = default;
    Int3(int x, int y = 0, int z = 0) 
        : x(x)
        , y(y)
        , z(z) {

    }
};

struct Float2 {
    float x;
    float y;

    Float2() = default;
    Float2(float x, float y = 0) 
        : x(x)
        , y(y) {

    }
};

struct Float3 {
    float x;
    float y;
    float z;

    Float3() = default;
    Float3(float x, float y = 0, float z = 0) 
        : x(x)
        , y(y)
        , z(z) {

    }
};

struct Float4 {
    float x;
    float y;
    float z;
    float w;

    Float4() = default;
    Float4(float x, float y = 0, float z = 0, float w = 0) 
        : x(x)
        , y(y)
        , z(z)
        , w(w) {

    }
};

typedef struct {
    unsigned short __x;
} half;
};