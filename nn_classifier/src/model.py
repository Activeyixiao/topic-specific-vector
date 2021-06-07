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




>>> from torch.nn.utils.rnn import pad_sequence
>>> a = torch.ones(25, 300)
>>> b = torch.ones(22, 300)
>>> c = torch.ones(15, 300)
>>> pad_sequence([a, b, c]).size()

class CNN_classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls = nn.Linear(int(config.in_features), 1, bias=True)
        self.sigmoid_fn = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.CNN = nn.Conv1d(in_channels=int(config.in_features), out_channels=20, kernel_size=1)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, input, target=None):
        logits = self.cls(input).squeeze(dim=1)
        probs = F.sigmoid(logits)
        lprobs = F.logsigmoid(logits)
        prob = -(lprobs * probs).sum()
        if target is not None:
            loss = self.loss_fn(prob, target)
            return prob, loss
        return  prob




def CNN_classifier(dim):
    model1 = Sequential()
    #model1.add(Dropout(0.2, input_shape=(dim,768)))
    model1.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(dim,768)))
    model1.add(GlobalMaxPooling1D())

    model2_in = Input(shape=(768,),name='orig_words_in')
    model2_out = Dense(36, activation='relu', name='orig_words_out')(model2_in)
    model2 = Model(model2_in, model2_out)
    Conca = Concatenate()([model1.output,model2.output])
    Result = Dense(1, activation='sigmoid')(Conca)
    NewModel = Model([model1.input,model2.input], Result) 
    return NewModel
    clf_example = CNN_classifier(5)
    print(clf_example.summary())