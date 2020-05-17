# Cuda 10.0 Base image
FROM nvidia/cuda:10.1-devel-ubuntu16.04 as mlbench-worker-base
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# CuDNN version that works with K80 and T4
ENV CUDNN_VERSION 7.5.0.56
ENV CUDA_VERSION 10.1
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# -------------------- CUDA DEPENDENCIES --------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda$CUDA_VERSION \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda$CUDA_VERSION && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    libc-dev \
    musl-dev \
    openssh-server \
    g++ \
    git \
    curl \
    sudo \
    iproute2

# -------------------- SSH --------------------
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

ARG SSH_USER=root
ENV SSH_USER=$SSH_USER
RUN mkdir -p /ssh-key/$SSH_USER && chown -R $SSH_USER:$SSH_USER /ssh-key/$SSH_USER
RUN mkdir -p /.sshd/host_keys && \
  chown -R $SSH_USER:$SSH_USER /.sshd/host_keys && chmod 700 /.sshd/host_keys
RUN mkdir -p /.sshd/user_keys/$SSH_USER && \
  chown -R $SSH_USER:$SSH_USER /.sshd/user_keys/$SSH_USER && chmod 700 /.sshd/user_keys/$SSH_USER
VOLUME /ssh-key/$SSH_USER


# -------------------- Conda environment --------------------
RUN curl -o ~/miniconda.sh -LO  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     sh ~/miniconda.sh -b -p /conda && rm ~/miniconda.sh
ENV PATH /conda/bin:$PATH
ENV LD_LIBRARY_PATH /conda/lib:$LD_LIBRARY_PATH

# TODO: Source code in Channel Anaconda can be outdated, switch to conda-forge if posible.
RUN conda install -y -c anaconda numpy pyyaml scipy mkl setuptools cmake cffi mkl-include typing \
    && conda install -y -c mingfeima mkldnn \
    && conda install -y -c soumith magma-cuda101 \
    && conda install -y -c conda-forge python-lmdb opencv numpy \
    && conda clean --all -y

# -------------------- Open MPI --------------------
RUN mkdir /.openmpi/
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz\
    && gunzip -c openmpi-4.0.3.tar.gz | tar xf - \
    && cd openmpi-4.0.3 \
    && ./configure --prefix=/.openmpi/ --with-cuda\
    && make all install \
    && rm /openmpi-4.0.3.tar.gz \
    && rm -rf /openmpi-4.0.3 \
    && apt-get remove -y wget

ENV PATH /.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH /.openmpi/lib:$LD_LIBRARY_PATH

RUN mv /.openmpi/bin/mpirun /.openmpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /.openmpi/bin/mpirun && \
    echo "/.openmpi/bin/mpirun.real" '--allow-run-as-root "$@"' >> /.openmpi/bin/mpirun && \
    chmod a+x /.openmpi/bin/mpirun

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /.openmpi/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /.openmpi/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /.openmpi/etc/openmpi-mca-params.conf

# configure the path.
RUN echo export 'PATH=$HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin:$HOME/.openmpi/bin:$PATH' >> ~/.bashrc
RUN echo export 'LD_LIBRARY_PATH=$HOME/.openmpi/lib:$LD_LIBRARY_PATH' >> ~/.bashrc


# -------------------- PyTorch --------------------
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Build pytorch from source
RUN git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch && \
    git checkout v1.5.0 && \
    git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST="3.5 3.7 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v . && \
    rm -rf /pytorch


RUN git clone https://github.com/pytorch/vision.git && cd vision && git checkout tags/v0.5.0 &&  pip install -v .
# RUN pip install -U git+https://github.com/ppwwyyxx/tensorpack.git

# RUN conda install -y -c anaconda msgpack
# RUN conda install -y -c anaconda msgpack msgpack-numpy pyzmq pillow
# RUN conda install -y -c conda-forge tqdm
# # RUN conda install -y -c pchrapka zmq
# # RUN conda install -c omnia termcolor

# -------------------- patch --------------------
# libGL.so.1 might be lost when nvidia driver is installed
# sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev
RUN apt-get install -y libgl1-mesa-glx

RUN pip install kubernetes dill

# -------------------- Others --------------------
RUN echo "orte_keep_fqdn_hostnames=t" >> /.openmpi/etc/openmpi-mca-params.conf

ADD ./entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/entrypoint.sh

# Copy your application code to the container (make sure you create a .dockerignore file if any large files or directories should be excluded)
RUN mkdir /app/
WORKDIR /app/

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/sbin/sshd","-eD", "-f", "/.sshd/user_keys/root/sshd_config"]
