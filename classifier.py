import torch 
import numpy as np

class MLP(torch.nn.Module):

    def __init__(self, input_size, ouput_size=1) -> None:
        super(MLP, self).__init__()
        self.FC1 = torch.nn.Linear(input_size, 16)
        self.FC2 = torch.nn.Linear(16, 8)
        self.FC3 = torch.nn.Linear(8, ouput_size)
        self.ReLU =  torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        out = self.ReLU(self.FC1(x))
        out = self.ReLU(self.FC2(out))
        out = self.softmax(self.FC3(out))
        return out


def init_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    return device



def train(datas,labels, model, criterion, optimizer, device='cpu'):
    model.train()
    datas = datas.to(device)
    labels = labels.to(device)
    labels = torch.unsqueeze(labels, 1)
    for i in range(100):
        predicted = model(datas)
        loss = criterion(predicted, labels)
        # print(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def validation(datas,labels, model, criterion, device):
    model.eval()
    datas = datas.to(device)
    labels = labels.to(device)
    labels = torch.unsqueeze(labels, 1)
    predicted = model(datas)
    loss = criterion(predicted, labels)
    acc = torch.mean((predicted==labels).float())
    return loss.item(), acc


def call_from_matlab(datas_train, datas_val, labels_train, labels_val, input_size, ouput_size):
    device = init_device()
    model = MLP(input_size, ouput_size).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    datas_train = torch.tensor(datas_train).float()
    datas_val = torch.tensor(datas_val).float()
    labels_train = torch.tensor(labels_train).float()
    labels_val = torch.tensor(labels_val).float()
    loss_init, acc_init = validation(datas_val, labels_val, model, criterion, device)
    print(loss_init, acc_init)
    train(datas_train,labels_train, model, criterion, optimizer, device='cpu')
    loss_final, acc_final = validation(datas_val, labels_val, model, criterion, device)
    print(loss_final, acc_final)
    