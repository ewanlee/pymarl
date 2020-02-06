FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
MAINTAINER Tabish Rashid

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# Python packages we use (or used at one point...)
RUN pip3 install numpy scipy pyyaml matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install imageio -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install tensorboard-logger -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install pygame -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install gym==0.9.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle==0.9.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install torchvision snakeviz pytest probscale -i https://pypi.tuna.tsinghua.edu.cn/simple

## -- SMAC
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /pymarl/3rdparty/StarCraftII

WORKDIR /pymarl
