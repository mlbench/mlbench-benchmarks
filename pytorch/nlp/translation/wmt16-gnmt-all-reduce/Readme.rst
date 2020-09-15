PyTorch WMT16 GNMT Machine Translation
""""""""""""""""""""""""""""""""""""""

GNMT implementation (adapted from `NVIDIA's implementation <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT>`_

:Task: :ref:`Task 4a <mlbench-docs:benchmark-task-4a>`
:Framework: PyTorch
:Communication Backend: Open MPI, NCCL and GLOO (PyTorch `torch.distributed`)
:Distribution Algorithm: All-Reduce
:Model: GNMT
:Dataset: WMT16
:GPU: Yes
:Seed: 42
:Image Location: mlbench/pytorch-wmt16-gnmt-all-reduce:latest