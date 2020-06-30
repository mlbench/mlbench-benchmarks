PyTorch WMT17 Transformer Machine Translation
"""""""""""""""""""""""""""""""""""""""""""""

Transformer implementation `Attention Is All You need <https://arxiv.org/abs/1706.03762>`_

:Task: `Task 4b <https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#b-transformer-wmt17-en-de>`_
:Framework: PyTorch
:Communication Backend: Open MPI, NCCL and GLOO (PyTorch `torch.distributed`)
:Distribution Algorithm: All-Reduce
:Model: Transformer
:Dataset: WMT17
:GPU: Yes
:Seed: 42
:Image Location: mlbench/pytorch-wmt17-transformer-all-reduce:latest
