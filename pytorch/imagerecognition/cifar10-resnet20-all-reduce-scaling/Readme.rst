PyTorch Cifar-10 ResNet-20 All-Reduce
"""""""""""""""""""""""""""""""""""""

Resnet 20 implementation for CIFAR-10 using All-Reduce

:Task: `Task 1a <https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-resnet-20-cifar-10>`_
:Framework: PyTorch
:Communication Backend: Open MPI, GLOO, NCCL (PyTorch `torch.distributed`)
:Distribution Algorithm: All-Reduce
:Model: ResNet-20
:Dataset: CIFAR-10
:GPU: Yes
:Seed: 42
:Image Location: mlbench/pytorch-cifar10-resnet-scaling:latest