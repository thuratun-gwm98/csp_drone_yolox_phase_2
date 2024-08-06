# FROM nvcr.io/nvidia/cuda:11.7.1-base-ubuntu22.04
FROM nvcr.io/nvidia/tensorrt:23.08-py3
# ARG BASE_IMG=nvidia/cuda:11.6.2-base-ubuntu20.04
# FROM ${BASE_IMG} as base
# ENV BASE_IMG=nvidia/cuda:11.6.2-base-ubuntu20.04

# ENV DEBIAN_FRONTEND=noninteractive
# ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# ARG TENSORRT_VERSION=8.5.3
# ENV TENSORRT_VERSION=${TENSORRT_VERSION}
# RUN test -n "$TENSORRT_VERSION" || (echo "No tensorrt version specified, please use --build-arg TENSORRT_VERSION=x.y to specify a version." && exit 1)

# ARG PYTHON_VERSION=3.7
# ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN apt-get update -y
RUN apt-get --fix-missing install -y wget build-essential checkinstall libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev libgl1-mesa-dev libsndfile1 libglib2.0-0
RUN apt-get install -y \
    g++ gcc cmake \
    git curl python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.0.1 torchvision==0.15.2
RUN pip3 install onnx==1.14.0 onnxruntime==1.15.1

WORKDIR /home/yolox-tensor
# COPY YOLOX CODES
COPY yolox/ ./yolox
COPY setup.py ./setup.py
COPY setup.cfg ./setup.cfg
COPY tools/ ./tools/
COPY exps/ ./exps/
COPY README.md ./README.md
COPY  requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -v -e .

RUN pip3 install tensorrt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
    && cd torch2trt \
    && python3 setup.py install

RUN pip3 install cython

