FROM mlbench/mlbench-pytorch-base:cuda-v10.0

# -------------------- Debug --------------------
# RUN apt-get update && apt-get install -y vim net-tools

# The reference implementation and user defined implementations are placed here.
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd ..

#RUN apt-get install -y debhelper dkms
## nv_peer_memory for faster NCCL
#RUN git clone https://github.com/Mellanox/nv_peer_memory.git && cd nv_peer_memory && \
#    ./build_module.sh && cd /tmp && tar xzf /tmp/nvidia-peer-memory_1.0.orig.tar.gz && \
#    cd nvidia-peer-memory-1.0 && dpkg-buildpackage -us -uc && dpkg -i nv-peer-memory-dkms_1.0-8_all.deb && \
#    cd /

# Horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod

ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN /conda/bin/python -m spacy download en && /conda/bin/python -m spacy download de

RUN mkdir /codes
ADD ./main.py /codes/main.py

ENV PYTHONPATH /codes
#ENV HOROVOD_CACHE_CAPACITY 2048
#ENV MPICH_MAX_THREAD_SAFETY multiple
# To find `libnvidia-ml.so` on google cloud.
# ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true