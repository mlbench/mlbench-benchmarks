FROM mlbench/mlbench-pytorch-base:pytorch-v1.8.1

# -------------------- Debug --------------------
# RUN apt-get update && apt-get install -y vim net-tools

# The reference implementation and user defined implementations are placed here.
ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir /codes
RUN mkdir /codes/utils

ADD main.py /codes/main.py
ADD ./utils/* /codes/utils/

ENV PYTHONPATH /codes

# To find `libnvidia-ml.so` on google cloud.
# ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true