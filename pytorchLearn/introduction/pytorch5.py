#可选: 数据并行处理
#我们将学习如何使用数据并行(DataParallel）来使用多GPU。
#PyTorch非常容易的就可以使用GPU，
#可以用如下方式把一个模型放到GPU上:


#device = torch.device("cuda: 0")
#model.to(device)

#复制所有的张量到GPU上:
#mytensor = my_tensor.to(device)
#调用my_tensor.to(device)返回一个GPU上的my_tensor副本，
#而不是重写my_tensor。
#把它赋值给一个新的张量
#并在GPU上使用这个张量。

#在多GPU上执行正向和反向传播是自然而然的事。
#PyTorch默认将只是用一个GPU。
#你可以使用DataParallel让模型并行运行来轻易的在多个GPU上运行你的操作。
#model = nn.DataParallel(model)
#这是这篇教程背后的核心

#导入和参数
#导入PyTorch模块和定义参数。
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Parameters和DataLoaders
input_size = 5
output_size = 2


batch_size = 30
data_size = 100

#设备(Device）:
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

#虚拟数据集
#要制作一个虚拟数据集，你只需实现__getitem__。
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size = batch_size, shuffle= True)

#简单模型
#作为演示，我们的模型只接受一个输入，执行一个线性操作，然后得到结果。
#在任何模型(CNN，RNN，Capsule Net等）上使用DataParallel
#在模型内部放置了一条打印语句来检测输入和输出向量的大小。
#请注意批等级为0时打印的内容。
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

#创建一个模型和数据并行
#这是本教程的核心部分。
#首先，我们需要创建一个模型实例和检测我们是否有多个GPU。
#如果我们有多个GPU，我们使用nn.DataParallel来包装我们的模型。
#然后通过model.to(device)把模型放到GPU上。
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
print("执行完毕")
model.to(device)

#运行模型
#现在我们可以看输入和输出张量的大小。
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("outSize: input size",input.size(),
          "output_size", output.size())

#结果
#如果没有GPU或只有1个GPU，
#当我们对30个输入和输出进行批处理时，我们和期望的一样得到30个输入和30个输出，
#但是若有多个GPU，会得到如下的结果。

#2个GPU
#若有2个GPU，将看到:

#3个GPU
#若有3个GPU，将看到:

#8个GPU
#若有8个GPU，将看到:

#总结
#DataParallel自动的划分数据，
#并将作业顺序发送到多个GPU上的多个模型。
#DataParallel会在每个模型完成作业后，
#收集与合并结果然后返回给你。



















