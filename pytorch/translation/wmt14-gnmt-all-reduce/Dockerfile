FROM mlbench/mlbench-pytorch-base:pytorch-v1.5.0

# -------------------- Debug --------------------
# RUN apt-get update && apt-get install -y vim net-tools

# The reference implementation and user defined implementations are placed here.
#RUN git clone https://github.com/NVIDIA/apex && cd apex && git checkout f3a960f && \
#    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
#    cd ..
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--xentropy" ./ && \
    cd ..

# Horovod
RUN HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_GPU_BROADCAST=MPI \
    HOROVOD_WITH_PYTORCH=1\
    HOROVOD_WITH_MPI=1\
    pip install --no-cache-dir horovod

RUN pip install spacy==2.2.3
RUN /conda/bin/python -m spacy download en && /conda/bin/python -m spacy download de

ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir /codes
RUN mkdir /codes/utils

ADD ./main.py /codes/main.py
ADD ./utils/* /codes/utils/

ENV PYTHONPATH /codes

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true