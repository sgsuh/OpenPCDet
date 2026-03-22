FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt update && \
    apt install -y libgl1 \
    libusb-1.0-0-dev \
    libpcl-dev \
    libyaml-cpp-dev

RUN pip install numpy==1.24.0 \
    opencv-python==4.8.0.74 \
    tensorboardX \
    easydict \
    pyquaternion \
    open3d \
    SharedArray \
    spconv-cu120 \
    kornia \
    av2==0.2.1 \
    onnxsim \
    onnx-graphsurgeon \
    scikit-image