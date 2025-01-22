from utils import Registry
import torch.nn as nn
import torch

from helper.optim import PolyLR

DATASETS = Registry('DATASETS')
TRANSFORMS = Registry('TRANSFORMS')
MODELS = Registry('MODELS')
LOSSES = Registry('LOSSES')

OPTIMIZER = Registry('OPTIMIZER')
LR_SHEDULER = Registry('LR_SHEDULER')

METRICS = Registry('METRICS')

LOSSES.register_module(name='SmoothL1Loss', module=nn.SmoothL1Loss)
LOSSES.register_module(name='MSELoss', module=nn.MSELoss)
LOSSES.register_module(name='HuberLoss', module=nn.HuberLoss)
LOSSES.register_module(name='BCEWithLogitsLoss', module=nn.BCEWithLogitsLoss)
OPTIMIZER.register_module(name='Adam', module=torch.optim.Adam)
OPTIMIZER.register_module(name='AdamW', module=torch.optim.AdamW)
LR_SHEDULER.register_module(name='PolyLR', module=PolyLR)