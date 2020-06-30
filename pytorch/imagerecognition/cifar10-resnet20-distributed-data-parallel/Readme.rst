PyTorch Cifar-10 ResNet-20 DDP
""""""""""""""""""""""""""""""

Resnet 20 implementation for CIFAR-10 using PyTorch DDP

:Task: `Task 1a <https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-resnet-20-cifar-10>`_
:Framework: PyTorch
:Communication Backend: NCCL (PyTorch `torch.distributed`)
:Distribution Algorithm: Distributed Data Parallel
:Model: ResNet-20
:Dataset: CIFAR-10
:GPU: Yes
:Seed: 42
:Image Location: mlbench/pytorch-cifar10-resnet-ddp:latest