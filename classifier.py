import torch 
import numpy as np
import random

class MLP(torch.nn.Module):

    def __init__(self, input_size, ouput_size=1) -> None:
        super(MLP, self).__init__()
        self.layer_1 = torch.nn.Linear(input_size, 64) 
        self.layer_2 = torch.nn.Linear(64, 32)
        self.layer_3 = torch.nn.Linear(32, 16)
        self.layer_out = torch.nn.Linear(16, ouput_size) 
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return x


def init_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    return device


def train(datas,labels, model, criterion, optimizer, datas_val, labels_val, device='cpu'):
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []

    epoch_num = 50
    datas = datas.to(device)
    labels = labels.to(device)
    labels = torch.unsqueeze(labels, 1)

    model.train()
    for epoch in range(epoch_num):
        optimizer.zero_grad()
        predicted = model(datas)

        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()

        predicted = model(datas)
        y_pred_tag = torch.round(torch.sigmoid(predicted))

        correct_results_sum = (y_pred_tag == labels).sum().float()
        acc = correct_results_sum/labels.shape[0]
        acc = torch.round(acc * 100)

        loss_train.append(loss.item())
        acc_train.append(acc.item())

        loss_val_tmp, acc_val_tmp= validation(datas_val,labels_val, model, criterion, device)

        loss_val.append(loss_val_tmp)
        acc_val.append(acc_val_tmp.item())

    return model, loss_train, loss_val, acc_train, acc_val


def validation(datas,labels, model, criterion, device):
    model.eval()
    datas = datas.to(device)
    labels = labels.to(device)
    labels = torch.unsqueeze(labels, 1)
    with torch.no_grad():
        predicted = model(datas)
    loss = criterion(predicted, labels)
    acc = torch.mean((predicted==labels).float())

    y_pred_tag = torch.round(torch.sigmoid(predicted))

    correct_results_sum = (y_pred_tag == labels).sum().float()
    acc = correct_results_sum/labels.shape[0]
    acc = torch.round(acc * 100)
    return loss.item(), acc


def call_from_matlab(datas_train, datas_val, labels_train, labels_val, input_size, ouput_size):

    device = init_device()
    model = MLP(input_size, ouput_size).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    datas_train = torch.tensor(datas_train).float()
    datas_val = torch.tensor(datas_val).float()
    labels_train = torch.tensor(labels_train).float()
    labels_val = torch.tensor(labels_val).float()

    model, loss_train, loss_val, acc_train, acc_val = train(datas_train,labels_train, model, criterion, optimizer, datas_val, labels_val, device='cpu')

    return loss_train, loss_val, acc_train, acc_val
    