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
import sys



class Dis_critic(torch.nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Dis_critic,self).__init__()
        self.fc_net = nn.Sequential(
            OrderedDict([
          ('dense1', nn.Linear(state_dim+action_dim,512)),
          ('norm1', nn.BatchNorm1d(512)),
          ('relu1', nn.ELU()),
          ("dense2", nn.Linear(512, 256)),
          ("norm2", nn.BatchNorm1d(256)),
          ("elu2", nn.ELU()),
          ("dense3", nn.Linear(256, 128)),
          ("norm3", nn.BatchNorm1d(128)),
          ("relu3", nn.ELU()),
          ("dense4", nn.Linear(128,64)),
          ("norm4", nn.BatchNorm1d(64)),
          ("elu4", nn.ELU()),
          ("dense5", nn.Linear(64,1))
          ])
        )
    def forward(self, x,a):
        if len(x.size())==3:
            x = torch.squeeze(x,1)
        # print("X shape is", x.size())
        # print("A shape is", a.size())
        return self.fc_net(torch.cat([x, a], 1))


class Hybrid_critic(torch.nn.Module):
    def __init__(self,num_area,hidden_dim):
        super(Hybrid_critic,self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agent=num_area
        self.hidden_dim=hidden_dim
        self.hyper_w1=hypernetworks(self.num_agent*self.hidden_dim)
        self.hyper_b1 = hypernetworks(self.hidden_dim)
        self.hyper_w2 = hypernetworks(self.hidden_dim)
        self.hyper_b2 = hypernetworks(1)

    def forward(self, agent_qs,g_s):
#        agent_qs=torch.tensor(agent_qs).view(-1,1,self.num_agent)
        W1 = torch.abs(self.hyper_w1(g_s).view(-1, self.num_agent, self.hidden_dim))
        b1 = self.hyper_b1(g_s).view(-1, 1, self.hidden_dim)
        # print("***********************")
        # print(torch.bmm(agent_qs, W1).shape)
        Q_total=F.relu(torch.bmm(agent_qs, W1) + b1)

        # print("****",Q_total.shape)

        W2 = torch.abs(self.hyper_w2(g_s).view(-1, self.hidden_dim,1))
        b2 = self.hyper_b2(g_s).view(-1, 1, 1)

        Q_tot = torch.bmm(Q_total, W2) + b2


        return Q_tot.view(-1,1)


class hypernetworks(torch.nn.Module):
    def __init__(self,output_dim):
        super(hypernetworks,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # branch:b*9*185
        self.conv1_b = nn.Conv1d(9, 16, kernel_size=3, stride=2, padding=1)
        # b*16*93
        self.pool1_b = nn.MaxPool1d(kernel_size=2, stride=2)
        # b*16*46
        self.conv2_b = nn.Conv1d(16, 8, kernel_size=2, stride=2, padding=1)
        # b*8*24
        self.pool2_b = nn.MaxPool1d(kernel_size=2, stride=1)
        # b*8*23=b*184

        # load:b*2*91
        self.conv1_d = nn.Conv1d(2, 4, kernel_size=3, stride=2, padding=1)
        # b*4*46
        self.pool1_d = nn.MaxPool1d(kernel_size=2, stride=2)
        # b*4*23=b*92

        self.fc_net = nn.Sequential(
            OrderedDict([
                ('dense1', nn.Linear(349, 512)),  # 1717
                ('norm1', nn.BatchNorm1d(512)),
                ('elu1', nn.ELU()),
                ("dense2", nn.Linear(512, 256)),
                ("norm2", nn.BatchNorm1d(256)),
                ("elu2", nn.ELU()),
                ("dense3", nn.Linear(256, 128)),
                ("norm3", nn.BatchNorm1d(128)),
                ("relu3", nn.ELU()),
                ("dense4", nn.Linear(128, output_dim))
            ])
        )

    def forward(self, x):
        br = x["branch"].view(-1, 9, 185)
        br = self.conv1_b(br)
        br = self.pool1_b(br)
        br = self.conv2_b(br)
        br = self.pool2_b(br)
        br = torch.flatten(br, start_dim=1)

        ld = x["load"].view(-1, 2, 91)
        ld = self.conv1_d(ld)
        ld = self.pool1_d(ld)
        ld = torch.flatten(ld, start_dim=1)

        el = x["else"].view(-1, 73)

        fc_input = torch.cat((br, ld, el), dim=1)

        # print("fc_input shape is", fc_input.size())
        return self.fc_net(fc_input)



# if __name__ == "__main__":
#     a=torch.rand(64,9,185)
#     b=torch.rand(64,2,91)
#     c=torch.rand(64,73)
#     g_s={"branch":a,"load":b,"else":c}
#     qs=torch.rand(64,1,10)
#
#     mn=Hybrid_critic(10,3)
#     print(mn(qs,g_s))




