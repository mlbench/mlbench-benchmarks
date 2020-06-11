FROM mlbench/mlbench-pytorch-base:pytorch-v1.5.0

RUN HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_GPU_BROADCAST=MPI \
    HOROVOD_WITH_PYTORCH=1\
    HOROVOD_WITH_MPI=1\
    pip install --no-cache-dir horovod

ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir /codes
ADD ./main.py /codes/main.py

ENV PYTHONPATH /codes
#ADD id_rsa /id_rsa
ENV NCCL_DEBUG INFO
ENV NCCL_DEBUG_SUBSYS ALL
#ENV NCCL_DEBUG_FILE /nccl.log
#ENV NCCL_IB_DISABLE 1
#
#RUN apt-get install -y libibverbs-dev

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true