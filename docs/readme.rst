.. |br| raw:: html

    <br>

MLBench Benchmarks
==================

MLBench contains several benchmark tasks and implementations. Tasks combinations of datasets and target metrics, whereas the implementations are concrete models and code that solve a task.

For an overview of MLBench tasks, please refer to the :doc:`Benchmarking Tasks Section <mlbench-docs:benchmark-tasks>`


Closed Division Benchmark Implementations
-----------------------------------------

A Benchmark Implementation is a model with fixed hyperparameters that solves a Benchmark Task.


Image Recognition
~~~~~~~~~~~~~~~~~


CIFAR-10
++++++++

.. table:: CIFAR-10 Benchmark Implementations. TtA @ 1, 2, 4, 8, 16 = Time to Accuracy with 1, 2, 4, 8, 16 workers. AaT 2h @ 1, 2, 4, 8, 16 = Accuracy after Time (2 Hours) with 1, 2, 4, 8, 16 workers
   :widths: auto

   +-------------------------------------+-----------+---------+-------------+-----------+------------------------------+------------------------------+--------------------------------------------+
   | Name                                | Framework | Backend | Parallelism | Model     | TtA @ 1, 2, 4, 8, 16         | AaT 2h @ 1, 2, 4, 8, 16      | Details                                    |
   +=====================================+===========+=========+=============+===========+==============================+==============================+============================================+
   | openmpi-cifar10-resnet18-all-reduce | PyTorch   | OpenMPI | All-Reduce  | ResNet-18 | 1: XX.X  |br|                | 1: XX.X |br|                 | :doc:`openmpi-cifar10-resnet18-all-reduce` |
   |                                     |           |         |             |           | 2: XX.X  |br|                | 2: XX.X |br|                 |                                            |
   |                                     |           |         |             |           | 4: XX.X  |br|                | 4: XX.X |br|                 |                                            |
   |                                     |           |         |             |           | 8: XX.X  |br|                | 8: XX.X |br|                 |                                            |
   |                                     |           |         |             |           | 16: XX.X |br|                | 16: XX.X  |br|               |                                            |
   +-------------------------------------+-----------+---------+-------------+-----------+------------------------------+------------------------------+--------------------------------------------+


ImageNet
++++++++

[.....]



All Implementations

.. toctree::

   openmpi-cifar10-resnet18-all-reduce