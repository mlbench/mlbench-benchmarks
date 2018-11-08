FROM mlbench-pytorch-base:latest

RUN pip install --extra-index-url https://testpypi.python.org/pypi mlbench-core

# The reference implementation and user defined implementations are placed here.
# ADD ./requirements.txt /requirements.txt
# RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir /codes
ADD ./train_dist.py /codes/train_dist.py

ENV PYTHONPATH /codes