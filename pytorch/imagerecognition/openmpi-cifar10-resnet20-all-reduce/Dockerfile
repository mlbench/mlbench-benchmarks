FROM  mlbench/mlbench-pytorch-base:pytorch-v1.0.0

# -------------------- Debug --------------------
RUN apt-get update && apt-get install -y vim net-tools

# The reference implementation and user defined implementations are placed here.
ADD ./mlbench_core-1.3.4-py3-none-any.whl /mlbench_core-1.3.4-py3-none-any.whl
RUN pip install --no-cache-dir /mlbench_core-1.3.4-py3-none-any.whl
ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir /codes
ADD ./main.py /codes/main.py

ENV PYTHONPATH /codes

# To find `libnvidia-ml.so` on google cloud.
# ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true
RUN pip install tensorpack
