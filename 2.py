import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

import torch.optim as optim

# 准备数据集
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #输⼊1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        #输⼊10通道，输出20通道，kernel 5*5
        self.conv2 = nn.Conv2d(10,20,5)
        #输⼊20通道，输出40通道，kernel 3*3
        self.conv3 = nn.Conv2d(20,40,3)
        # 2*2的池化层
        self.mp = nn.MaxPool2d(2)
        #全连接层//输⼊特征数，输出
        self.fc = nn.Linear(40,10)

    def forward(self, x):
        # in_size = 64
        # one batch此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x， y)
        # x.size(0)是指batchsize的值，把batchsize的值作为⽹络的in_size
        in_size = x.size(0)
        # x: 64*1*28*28
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*10*12*12  (n+2p-f)/s + 1 = 28 - 5 + 1 = 24,所以在没有池化的时候是24*24,池化层为2*2，所以池化之后为12*12
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*20*4*4同理，没有池化的时候是12 - 5 + 1 = 8，池化后为4*4
        x = F.relu(self.mp(self.conv3(x)))
        #输出x : 64*40*2*2
        x = x.view(in_size,-1)#平铺tensor相当于resharp
        # print(x.size())
        # x: 64*320
        x = self.fc(x)
        # x: 64*10
        # print(x.size())
        return F.log_softmax(x)#64*10


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs,target=inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%.5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            inputs,target=data
            inputs,target=inputs.to(device),target.to(device)
            outputs=model(inputs)
            _,predicted=torch.max(outputs.data,dim=1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()
    print('Accuracy on test set:%d %% [%d%d]' %(100*correct/total,correct,total))

if __name__ =='__main__':
    for epoch in range(100):
        train(epoch)
        test()