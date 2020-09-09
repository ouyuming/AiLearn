import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #输入图像channel：1；输出channel：6；5X5卷积核
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        #an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2X2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #如果是方阵，则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] #除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
#print(net)

#我们只需要定义 forward 函数，
#backward函数会在使用autograd时自动定义，
#backward函数用来计算导数。
#我们可以在 forward 函数中使用任何针对张量的操作和计算。
#一个模型的可学习参数可以通过net.parameters()返回

params = list(net.parameters())
#print(len(params))
#print(params[0].size())

#让我们尝试一个随机的32x32的输入。
#注意:这个网络(LeNet）的期待输入是32x32的张量。
#如果使用MNIST数据集来训练这个网络，要把图片大小重新调整到32x32。
input = torch.rand(1, 1, 32, 32)
out = net(input)
#print(out)

#清零所有参数的梯度缓存，然后进行随机梯度的反向传播：
net.zero_grad()
out.backward(torch.rand(1, 10))

#注意
#torch.nn只支持小批量处理(mini-batches）
#整个torch.nn包只支持小批量样本的输入，不支持单个样本的输入。
#比如，nn.Conv2d 接受一个4维的张量，即nSamples x nChannels x Height x Width
#如果是一个单独的样本，只需要使用input.unsqueeze(0)来添加一个“假的”批大小维度。

#复习目前为止看到的所有类。
#torch.Tensor - 一个多维数组，支持诸如backward()等的自动求导操作，同时也保存了张量的梯度。

#nn.Module - 神经网络模块。
#是一种方便封装参数的方式，具有将参数移动到GPU、导出、加载等功能。

#nn.Parameter  - 张量的一种，
#当它作为一个属性分配给一个Module时，它会被自动注册为一个参数。

#autograd.Function - 实现了自动求导前向和反向传播的定义
#每个Tensor至少创建一个Function节点，该节点连接到创建Tensor的函数并对其历史进行编码。

#损失函数
#一个损失函数接受一对(output, target)作为输入，计算一个值来估计网络的输出和目标值相差多少。
#nn包中有很多不同的损失函数。
#nn.MSELoss是比较简单的一种，它计算输出和目标的均方误差(mean-squared error）。
output = net(input)
target = torch.rand(10) #本例子中使用模拟数据
target = target.view(1, -1) #使目标值与数据值尺寸一致
criterion = nn.MSELoss()

loss = criterion(output, target)
#print(loss)

#现在，如果使用loss的.grad_fn属性跟踪反向传播过程，会看到计算图如下：
#input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#      -> view -> linear -> relu -> linear -> relu -> linear
#      -> MSELoss
#     -> loss

#所以，当我们调用loss.backward()，整张图开始关于loss微分，
#图中所有设置了requires_grad=True的张量的.grad属性累积着梯度张量。
#为了说明这一点，让我们向后跟踪几步：
#print(loss.grad_fn) #MSELoss
#print(loss.grad_fn.next_functions[0][0]) #Linear
#print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLU

#反向传播
#调用loss.backward()来反向传播误差
#我们需要清零现有的梯度，否则梯度将会与已有的梯度累加。
#我们将调用loss.backward()，并查看conv1层的偏置(bias）在反向传播前后的梯度。

net.zero_grad() #清零所有参数（parameter）的梯度缓存

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward() #损失函数的使用

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


#更新权重
#最简单的更新规则是随机梯度下降法(SGD）:
# weight = weight - learning_rate * gradient
#代码实现
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

#在使用神经网络时，可能希望使用各种不同的更新规则
#如SGD、Nesterov-SGD、Adam、RMSProp等
#为此，我们构建了一个较小的包torch.optim，它实现了所有的这些方法。使用它很简单：

import torch.optim as optim

#创建优化器（optimizer）
optimizer = optim.SGD(net.parameters(), lr=0.01)

#在训练的迭代中：
optimizer.zero_grad() #清零梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() #更新参数

#观察梯度缓存区是如何使用optimizer.zero_grad()手动清零的。
#这是因为梯度是累加的，正如前面反向传播章节叙述的那样

































