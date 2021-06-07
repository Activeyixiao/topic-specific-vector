import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))

class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls = nn.Linear(int(config.in_features), 1, bias=True)
        self.sigmoid_fn = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input, target=None):
        logits = self.cls(input).squeeze(dim=1)
        probs = F.sigmoid(logits)
        lprobs = F.logsigmoid(logits)
        prob = -(lprobs * probs).sum()
        if target is not None:
            loss = self.loss_fn(prob, target)
            return prob, loss
        return  prob


class Binary_topic_Net(nn.Module):
    def __init__(self,config,num_layers):
        super().__init__()
        self.attention = nn.Parameter(torch.ones(num_layers),requires_grad=True)
        self.sigmoid_fn = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.cls = nn.Linear(int(config.in_features), 1, bias=True)

    def forward(self,x,mask,target=None):
        softm = nn.Softmax(dim=1)
        relu = nn.ReLU()
        weights = torch.mul(self.attention,mask)
        weights = softm(relu(weights))
        vector = x * weights[:, :,None]
        vector = torch.sum(vector,1)
        vector = F.normalize(vector,dim=1)
        logits = self.cls(vector).squeeze(dim=1) #torch.dot(vector,self.weight)
        probs = F.sigmoid(logits)
        lprobs = F.logsigmoid(logits)
        prob = -(lprobs * probs).sum()
        if target is not None:
            loss = self.loss_fn(prob, target)
            return prob, loss
        return  prob
