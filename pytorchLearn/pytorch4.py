#训练分类器
#定义网络、计算损失、更新网络的权重

#数据应该怎么办呢？
#通常来说，当必须处理图像、文本、音频或视频数据时，
#可以使用python标准库将数据加载到numpy数组里。
#然后将这个数组转化成torch.*Tensor
##对于图片，有Pillow，OpenCV等包可以使用
##对于音频，有scipy和librosa等包可以使用
##对于文本，不管是原生python的或者是基于Cython的文本，可以使用NLTK和SpaCy
#特别对于视觉方面，我们创建了一个包，名字叫torchvision，
#其中包含了针对Imagenet、CIFAR10、MNIST等常用数据集的数据加载器(data loaders），
#还有对图像数据转换的操作，即torchvision.datasets和torch.utils.data.DataLoader。

#在这个教程中，我们将使用CIFAR10数据集，
#它有如下的分类：“飞机”，“汽车”，“鸟”，“猫”，“鹿”，“狗”，“青蛙”，“马”，“船”，“卡车”等。
#在CIFAR-10里面的图片数据大小是3x32x32，即：三通道彩色图像，图像大小是32x32像素。

#训练一个图片分类器
#我们将按顺序做以下步骤：
#1.通过torchvision加载CIFAR10里面的训练和测试数据集，并对数据进行标准化
#2.定义卷积神经网络
#3.定义损失函数
#4.利用训练数据训练网络
#5.利用测试数据测试网络

#1.加载并标准化CIFAR10
#使用torchvision加载CIFAR10超级简单。
import torch
import torchvision
import torchvision.transforms as transforms

#torchvision数据集加载完后的输出是范围在[0, 1]之间的PILImage。
#我们将其标准化为范围在[-1, 1]之间的张量。


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])    #这里打少了一对括号

#问题定位：这里少了dataset
#trainset = torchvision.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) #测试集
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#现在让我们可视化部分训练数据。
import matplotlib.pyplot as plt
import numpy as np


#输出图像的函数
def imshow(img):
    img = img / 2 + 0.5     #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1, 2, 0)))
    plt.show()

#随机获取训练图片
dataiter = iter(trainloader)
images,lables = dataiter.next()

#显示图片
imshow(torchvision.utils.make_grid(images))
#打印图片标签，随机生成
#print(' '.join('%5s' % classes[lables[j]] for j in range(4)))

#2.定义一个卷积神经网络
#将之前神经网络章节定义的神经网络拿过来，并将其修改成输入为3通道图像
#(替代原来定义的单通道图像）
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #输入图像channel：1；输出channel：6；5X5卷积核
        self.conv1 = nn.Conv2d(3,6,5) #改为3通道图像
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6,16,5)
        #an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # pooling
        x = self.pool(F.relu(self.conv1(x)))
        #如果是方阵，则可以只使用一个数字进行定义
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

#3.定义损失函数和优化器
#我们使用多分类的交叉熵损失函数和随机梯度下降优化器(使用momentum）
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#4.训练网络
#我们只需要遍历我们的数据迭代器，并将输入“喂”给网络和优化函数。

for epoch in range(2): #loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #get the inputs
        inputs, lables = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    #print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0


print('Finished Training')

#保存已训练得到的模型：
PATH = '../cifar_net.pth'
#torch.save(net.state_dict(),PATH)


#5.使用测试数据测试网络
#训练集上训练了2遍网络。
#检查网络是否学到了一些东西。
#通过预测神经网络输出的标签来检查这个问题
#并和正确样本进行(ground-truth）对比
#如果预测是正确的，我们将样本添加到正确预测的列表中。

#第一步。让我们展示测试集中的图像来熟悉一下。
dataiter = iter(testloader)
images, lables = dataiter.next()

#输出图片
#imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ',''.join('%5s' % classes[lables[j]] for j in  range(4)))

#第二步：加载保存的模型
#注意：在这里保存和加载模型不是必要的，我们只是为了解释如何去做这件事
net = Net()
net.load_state_dict(torch.load(PATH))

#让我们看看神经网络认为上面的例子是:
outputs = net(images)

#输出是10个类别的量值。
#一个类的值越高，网络就越认为这个图像属于这个特定的类。
#让我们得到最高量值的下标/索引；

_, predicted = torch.max(outputs, 1)
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

#网络在整个数据集上表现的怎么样
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, lables = data
        outputs = net(images)
        _,predicted = torch.max(outputs.data, 1)
        total += lables.size(0)
        correct += (predicted == lables).sum().item()

#print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#这比随机选取(即从10个类中随机选择一个类，正确率是10%）要好很多。
#看来网络确实学到了一些东西。
#那么哪些是表现好的类呢？哪些是表现的差的类呢？
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, lables = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == lables).squeeze()
        for i in range(4):
            lable = lables[i]
            class_correct[lable] += c[i].item()
            class_total[lable] += 1


#for i in range(10):
    #print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


#在GPU上训练
#与将一个张量传递给GPU一样，可以这样将神经网络转移到GPU上。
#如果我们有cuda可用的话，让我们首先定义第一个设备为可见cuda设备：
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

#本节的其余部分假设device是CUDA。
#这些方法将递归遍历所有模块，并将它们的参数和缓冲区转换为CUDA张量：
net.to(device)

#将输入和目标在每一步都送入GPU：
inputs, lables = inputs.to(device), lables.to(device)

#为什么我们感受不到与CPU相比的巨大加速？因为我们的网络实在是太小了。
#加宽你的网络
#注意第一个nn.Conv2d的第二个参数
#第二个nn.Conv2d的第一个参数要相同
#看看能获得多少加速。

#已实现的目标：
#在更高层次上理解PyTorch的Tensor库和神经网络
#训练一个小的神经网络做图片分类






















