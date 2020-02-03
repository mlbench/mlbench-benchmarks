# -*- coding: utf-8 -*-
from pcode.optim.sgd import SGD

from pcode.optim.dgc import DGC
from pcode.optim.parallel_choco import ParallelCHOCO

from pcode.optim.dcd_psgd import DCD_PSGD
from pcode.optim.ecd_psgd import ECD_PSGD


def define_optimizer(conf, model):
    # define the param to optimize.
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": conf.weight_decay if "bn" not in key else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]

    # define the optimizer.
    if conf.optimizer == "sgd":
        optim_class = SGD
    elif conf.optimizer == "dgc":
        optim_class = DGC
    elif conf.optimizer == "dcd_psgd":
        optim_class = DCD_PSGD
    elif conf.optimizer == "ecd_psgd":
        optim_class = ECD_PSGD
    elif conf.optimizer == "parallel_choco":
        optim_class = ParallelCHOCO
    else:
        raise NotImplementedError

    return optim_class(
        params,
        lr=conf.learning_rate,
        momentum=conf.momentum_factor,
        nesterov=conf.use_nesterov,
        conf=conf,
    )
