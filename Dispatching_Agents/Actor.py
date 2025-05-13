import csv
import random
import torch.utils.data as Data
import numpy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import  OrderedDict
from torch import optim
from torch.autograd import Variable
import os




class Dis_actor(torch.nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Dis_actor,self).__init__()

        self.fc_net = nn.Sequential(
            OrderedDict([
          ('dense1', nn.Linear(state_dim,512)),
          ('norm1', nn.BatchNorm1d(512)),
          ('elu1', nn.ELU()),
          ("dense2", nn.Linear(512, 256)),
          ("norm2", nn.BatchNorm1d(256)),
          ("elu2", nn.ELU()),
          ("dense3", nn.Linear(256, 128)),
          ("norm3", nn.BatchNorm1d(128)),
          ("elu3", nn.ELU()),
          ("dense4", nn.Linear(128,action_dim))
        ])
        )

    def forward(self,x):
        if len(x.size())==3:
            x = torch.squeeze(x,1)
        # print("X shape is", x.size())
        return self.fc_net(x)
