import csv
import random
import torch.utils.data as Data
import numpy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import  OrderedDict
from utilize.settings import settings
from torch import optim
from torch.autograd import Variable
import os



class adversarial_agent(torch.nn.Module):
    def __init__(self):
        super(adversarial_agent,self).__init__()
        self.device = torch.device('cuda')
        #branch:b*9*185
        self.conv1_b = nn.Conv1d(9, 16, kernel_size=3, stride=2, padding=1)
        #b*16*93
        self.pool1_b = nn.MaxPool1d(kernel_size=2, stride=2)
        #b*16*46
        self.conv2_b = nn.Conv1d(16, 8, kernel_size=2, stride=2, padding=1)
        #b*8*24
        self.pool2_b = nn.MaxPool1d(kernel_size=2, stride=1)
        # b*8*23=b*184

        # load:b*2*91
        self.conv1_d = nn.Conv1d(2, 4, kernel_size=3, stride=2, padding=1)
        # b*4*46
        self.pool1_d = nn.MaxPool1d(kernel_size=2, stride=2)
        # b*4*23=b*92


        self.fc_net = nn.Sequential(
            OrderedDict([
          ('dense1', nn.Linear(349,512)),#1717
          ('norm1', nn.BatchNorm1d(512)),
          ('elu1', nn.ELU()),
          ("dense2", nn.Linear(512, 256)),
          ("norm2", nn.BatchNorm1d(256)),
          ("elu2", nn.ELU()),
          ("dense3", nn.Linear(256, 128)),
          ("norm3", nn.BatchNorm1d(128)),
          ("relu3", nn.ELU()),
          ("dense4", nn.Linear(128,settings.num_attackable_branch))
        ])
        )

    def forward(self,x):
        br=x["branch"].view(-1, 9,185)
        br=self.conv1_b(br)
        br=self.pool1_b(br)
        br=self.conv2_b(br)
        br=self.pool2_b(br)
        br=torch.flatten(br, start_dim=1)

        ld=x["load"].view(-1, 2,91)
        ld=self.conv1_d(ld)
        ld=self.pool1_d(ld)
        ld = torch.flatten(ld, start_dim=1)

        el = x["else"].view(-1, 73)

        fc_input=torch.cat((br,ld,el),dim=1)

        # print("fc_input shape is", fc_input.size())
        return F.softmax(self.fc_net(fc_input),dim=1)









