PyTorch Wikitext2 AWD-LSTM Language Modeling
""""""""""""""""""""""""""""""""""""""""""""

AWD-LSTM Implementation for language Modeling in Wikitext2 dataset.
Model implementation taken from `SalesForce <https://github.com/salesforce/awd-lstm-lm>`_

:Task: :ref:`Task 3a <mlbench-docs:benchmark-task-3a>`
:Framework: PyTorch
:Communication Backend: Open MPI, NCCL and GLOO (PyTorch `torch.distributed`)
:Distribution Algorithm: All-Reduce
:Model: AWD-LSTM
:Dataset: Wikitext2
:GPU: Yes
:Seed: 43
:Image Location: mlbench/pytorch-wikitext2-lstm-all-reduce:latest