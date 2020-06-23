.. |br| raw:: html

    <br>
.. _benchmark-implementations:

MLBench Implementations
=======================

MLBench consists of several benchmark tasks and implementations. For each task on a dataset and target metric, we provide a reference implementation, as well as optional additional implementation variants for comparisons.

For an overview of all MLBench tasks, please refer to the :doc:`Benchmark Tasks Documentation <mlbench-docs:benchmark-tasks>`



Image Recognition
~~~~~~~~~~~~~~~~~


1a. ResNet, CIFAR-10
++++++++++++++++++++


.. include:: ../pytorch/imagerecognition/openmpi-cifar10-resnet20-all-reduce/Readme.rst

.. include:: ../pytorch/imagerecognition/openmpi-cifar10-resnet20-all-reduce-scaling/Readme.rst

.. include:: ../pytorch/linearmodels/openmpi-epsilon-logistic-regression-all-reduce/Readme.rst

.. include:: ../tensorflow/imagerecognition/openmpi-cifar10-resnet20-all-reduce/Readme.rst

