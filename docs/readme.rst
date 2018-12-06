.. |br| raw:: html

    <br>
.. _benchmark-implementations:

MLBench Benchmark Implementations
=================================

MLBench contains several benchmark tasks and implementations. Tasks combinations of datasets and target metrics, whereas the implementations are concrete models and code that solve a task.

For an overview of MLBench tasks, please refer to the :doc:`Benchmarking Tasks Section <mlbench-docs:benchmark-tasks>`


Closed Division Benchmark Implementations
-----------------------------------------

A Benchmark Implementation is a model with fixed hyperparameters that solves a Benchmark Task.


Image Recognition
~~~~~~~~~~~~~~~~~


1a. ResNet, CIFAR-10
++++++++++++++++++++

PyTorch Cifar-10 ResNet-20 Open-MPI
"""""""""""""""""""""""""""""""""""

:Framework: PyTorch
:Communication Backend: Open MPI (PyTorch `torch.distributed`)
:Distribution Algorithm: All-Reduce
:Model: ResNet-20
:Dataset: CIFAR-10
:GPU: Yes
:Seed: 42
:Image Location: /pytorch/imagerecognition/openmpi-cifar10-resnet20-all-reduce/

Tensorflow Cifar-10 ResNet-20 Open-MPI
""""""""""""""""""""""""""""""""""""""

.. TODO We use OpenMPI for starting processes, but communication is gRPC? document this more
   Also, All_Reduce is not exactly all-reduce. We need to comment on this

:Framework: TensorFlow
:Communication Backend: Open MPI
:Distribution Algorithm: All-Reduce
:Model: ResNet-20
:Dataset: CIFAR-10
:GPU: Yes
:Seed: 42
:Image Location: /tensorflow/imagerecognition/openmpi-cifar10-resnet20-all-reduce/
