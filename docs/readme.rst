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


.. include:: ../pytorch/imagerecognition/openmpi-cifar10-resnet20-all-reduce/Readme.rst

.. include:: ../pytorch/imagerecognition/openmpi-cifar10-resnet20-all-reduce-scaling/Readme.rst

.. include:: ../tensorflow/imagerecognition/openmpi-cifar10-resnet20-all-reduce/Readme.rst

