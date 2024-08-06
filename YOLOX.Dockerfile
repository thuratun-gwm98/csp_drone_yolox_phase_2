FROM nvcr.io/nvidia/cuda:11.6.2-base-ubuntu20.04
# FROM nvcr.io/nvidia/tensorrt:24.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs


# ARG PYTHON_VERSION=3.7
# ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN apt-get update -y
RUN apt-get --fix-missing install -y wget build-essential checkinstall libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev libgl1-mesa-dev libsndfile1 libglib2.0-0
RUN apt-get install -y \
    g++ gcc cmake \
    git curl \
    python3.7 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Python
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get install software-properties-common
# RUN apt-get install python3.7 -y
# RUN apt-get -y install python3-pip
# # Update symlink to point to latest
# RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.7 /usr/bin/python3

RUN pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

WORKDIR /home/yolox
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
