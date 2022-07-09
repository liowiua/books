from torchvision import datasets, transforms
# batch_size是指每次送⼊⽹络进⾏训练的数据量
batch_size =64
# MNIST Dataset
# MNIST数据集已经集成在pytorch datasets中，可以直接调⽤
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# import adabound


train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,)
class Net(nn.Module):
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
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
def train(epoch):
# 枚举、列举，对于⼀个可迭代/遍历的对象， enumerate将其组成⼀个索引序列，利⽤它可以同时获得索引和值
    for batch_idx,(data, target)in enumerate(train_loader):#batch_idx是enumerate（）函数⾃带的索引，从0开始
        # data.size():[64, 1, 28, 28]
        # target.size():[64]
        enumerate
        output = model(data)
        # output:64*10
        loss = F.nll_loss(output, target)
        #每200次，输出⼀次数据
        # if batch_idx %200==0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
        #         format(
        #         epoch,
        #         batch_idx *len(data),
        #         len(train_loader.dataset),
        #         100.* batch_idx /len(train_loader),
        #         loss.item()))
        optimizer.zero_grad()#所有参数的梯度清零
        loss.backward()#即反向传播求梯度
        optimizer.step()#调⽤optimizer进⾏梯度下降更新参数



# #实验⼊⼝
# for epoch in range(1,10):
#     train(epoch)

def test():
    test_loss =0
    correct =0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data.cuda())
        #累加loss
        test_loss += F.nll_loss(output, target.cuda(), size_average=False).item()
        # get the index of the max log-probability
        #找出每列（索引）概率意义下的最⼤值

        pred = output.data.max(1, keepdim=True)[1]
        # print(pred)
        correct += pred.eq(target.data.view_as(pred).cuda()).cuda().sum()

    test_loss /=len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct,len(test_loader.dataset),
        100.* correct /len(test_loader.dataset)))
#实验⼊⼝
for epoch in range(1,10):
    print("test num"+str(epoch))
    train(epoch)
    test()