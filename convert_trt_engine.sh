#!/bin/bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=./cpp/model/pointpillar.onnx \
    --fp16 \
    --plugins=./cpp/build/libpointpillar_core.so \
    --saveEngine=./cpp/model/pointpillar.plan \
    --inputIOFormats=fp16:chw,int32:chw,int32:chw \
    --verbose \
    --dumpLayerInfo \
    --dumpProfile \
    --separateProfileRun \
    --profilingVerbosity=detailed > ./cpp/model/pointpillar.8611.log 2>&1