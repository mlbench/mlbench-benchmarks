FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 as mlbench-worker-base
# TODO: reduce size and complexity of image.

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

# -----–––---------------------- Cuda Dependency --------------------
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        --allow-change-held-packages \
         libnccl2=2.0.5-3+cuda9.0 \
         libnccl-dev=2.0.5-3+cuda9.0 &&\
     rm -rf /var/lib/apt/lists/*

# -------------------- Conda environment --------------------
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh  && \
     sh ~/miniconda.sh -b -p /conda && rm ~/miniconda.sh
ENV PATH /conda/bin:$PATH
ENV LD_LIBRARY_PATH /conda/lib:$LD_LIBRARY_PATH

# TODO: Source code in Channel Anaconda can be outdated, switch to conda-forge if posible.
RUN conda install -y -c anaconda numpy pyyaml scipy mkl setuptools cmake cffi mkl-include typing \
    && conda install -y -c mingfeima mkldnn \
    && conda install -y -c soumith magma-cuda90 \
    && conda install -y -c conda-forge python-lmdb opencv numpy \
    && conda clean --all -y

# -------------------- Open MPI --------------------
RUN mkdir /.openmpi/
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz\
    && gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=/.openmpi/ --with-cuda\
    && make all install \
    && rm /openmpi-3.0.0.tar.gz \
    && rm -rf /openmpi-3.0.0 \
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

RUN conda install -y -c conda-forge mpi4py

# -------- Build Tensorflow with MPI support------
# source install instructions https://www.tensorflow.org/install/source
# pip six numpy wheel setuptools already installed
RUN pip install mock && \
    pip install keras_applications==1.0.6 --no-deps && \
    pip install keras_preprocessing==1.0.5 --no-deps

# install bazel
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add - && \
    apt-get update && apt-get -y install bazel && \
    apt-get install --only-upgrade bazel

# clone tensorflow repo
WORKDIR /tmp
RUN git clone https://github.com/tensorflow/tensorflow.git

# configure for CUDA 9.0 with mpi
WORKDIR /tmp/tensorflow
RUN echo '\n''\n'n'\n''\n'N'\n'y'\n'9'\n''\n''\n''\n'N'\n''\n''\n'N'\n''\n'y'\n''\n''\n'N'\n' | ./configure

# build package with bazel
RUN bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# install with pip
RUN TFLOW_PKG_VER=$(ls /tmp/tensorflow_pkg/) && \
    pip install /tmp/tensorflow_pkg/$TFLOW_PKG_VER

# -------------------- Others --------------------
RUN echo "orte_keep_fqdn_hostnames=t" >> /.openmpi/etc/openmpi-mca-params.conf

ADD ./entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/entrypoint.sh

# Copy your application code to the container (make sure you create a .dockerignore file if any large files or directories should be excluded)
RUN mkdir /app/
WORKDIR /app/
RUN git clone https://github.com/mlbench/mlbench-benchmarks.git
RUN pip install -r /app/mlbench-benchmarks/tensorflow/imagerecognition/openmpi-cifar10-resnet20-all-reduce/requirements.txt

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/sbin/sshd","-eD", "-f", "/.sshd/user_keys/root/sshd_config"]
