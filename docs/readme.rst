.. |br| raw:: html

    <br>
.. _benchmark-implementations:

=================================
MLBench Benchmark Implementations
=================================

MLBench consists of several benchmark tasks and implementations. For each task on a dataset and target metric, we provide a reference implementation, as well as optional additional implementation variants for comparisons.

For an overview of all MLBench tasks, please refer to the :doc:`Benchmark Tasks Documentation <mlbench-docs:benchmark-tasks>`

A Benchmark Implementation is a model with fixed hyperparameters that solves a Benchmark Task.

Task 0: Communication Backend
-----------------------------

This task is a dummy task that allows for testing the communication backends for various operations and frameworks.

0.a PyTorch All-reduce
^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../pytorch/backend_benchmark/Readme.rst

Task 1: Image Classification
----------------------------

1a. Resnet-20, CIFAR-10
^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../pytorch/imagerecognition/cifar10-resnet20-all-reduce/Readme.rst

.. include:: ../pytorch/imagerecognition/cifar10-resnet20-distributed-data-parallel/Readme.rst

1b. Resnet-?, ImageNet
^^^^^^^^^^^^^^^^^^^^^^
TODO

Task 2: Linear Learning
-----------------------

2.a Logistic Regression, Epsilon 2008
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. include:: ../pytorch/linearmodels/epsilon-logistic-regression-all-reduce/Readme.rst


Task 3: Language Modelling
--------------------------

3.a RNN, Wikitext2
^^^^^^^^^^^^^^^^^^

.. include:: ../pytorch/nlp/language-modeling/wikitext2-lstm-all-reduce/Readme.rst

Task 4: Machine Translation
---------------------------

4.a LSTM, WMT16 EN-DE
^^^^^^^^^^^^^^^^^^^^^

.. include:: ../pytorch/nlp/translation/wmt16-gnmt-all-reduce/Readme.rst

4.b Transformer, WMT17 EN-DE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../pytorch/nlp/translation/wmt17-transformer-all-reduce/Readme.rst