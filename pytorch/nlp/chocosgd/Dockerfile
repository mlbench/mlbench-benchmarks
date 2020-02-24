FROM  mlbench/mlbench-pytorch-base:pytorch-v1.0.0

# -------------------- Debug --------------------
RUN apt-get update && apt-get install -y vim net-tools

# The reference implementation and user defined implementations are placed here.
ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN /conda/bin/python -m spacy download en
RUN pip install --upgrade git+https://github.com/pytorch/text

# install bit2byte.
RUN git clone https://github.com/tvogels/signSGD-with-Majority-Vote.git && \
    cd signSGD-with-Majority-Vote/main/bit2byte-extension/ && \
    /conda/bin/python setup.py develop --user

RUN mkdir /codes
COPY . /codes/

ENV PYTHONPATH /codes

# To find `libnvidia-ml.so` on google cloud.
# ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true
