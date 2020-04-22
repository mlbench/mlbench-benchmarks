FROM mlbench/mlbench-pytorch-base:pytorch-v1.4.0

# -------------------- Debug --------------------
# RUN apt-get update && apt-get install -y vim net-tools

# The reference implementation and user defined implementations are placed here.
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd ..

ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN /conda/bin/python -m spacy download en && /conda/bin/python -m spacy download de

RUN mkdir /codes
ADD ./main.py /codes/main.py

ENV NCCL_DEBUG INFO
ENV PYTHONPATH /codes

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true