FROM mlbench/mlbench-pytorch-base:pytorch-v1.8.1

# The reference implementation and user defined implementations are placed here.
ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir /codes
ADD ./main.py /codes/main.py

ENV PYTHONPATH /codes

# To find `libnvidia-ml.so` on google cloud.
# ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true