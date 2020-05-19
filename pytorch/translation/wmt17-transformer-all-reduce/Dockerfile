FROM mlbench/mlbench-pytorch-base:pytorch-v1.4.0

# -------------------- Debug --------------------
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--xentropy" ./ && \
    cd ..

## Horovod
RUN HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_GPU_BROADCAST=MPI \
    HOROVOD_WITH_PYTORCH=1\
    HOROVOD_WITH_MPI=1\
    pip install --no-cache-dir horovod
#
#RUN apt-get -y install cmake
#
#RUN git clone https://github.com/NVIDIA/cutlass.git && cd cutlass && git checkout tags/v1.2.0 && git submodule update --init --recursive && \
#    mkdir build && cd build && cmake .. && make -j4

ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir /codes
RUN mkdir /codes/utils
#RUN mkdir /codes/padded_softmax
#RUN mkdir /codes/strided_batched_gemm

#ADD ./padded_softmax/* /codes/padded_softmax/
#ADD ./strided_batched_gemm/* /codes/strided_batched_gemm/

#ADD setup.py /codes/setup.py
#RUN cd /app/ && python /codes/setup.py build

ADD ./main.py /codes/main.py
ADD ./utils/* /codes/utils/







ENV PYTHONPATH /codes

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true