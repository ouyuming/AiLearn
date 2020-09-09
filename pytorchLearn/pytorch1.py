from __future__ import print_function
import torch

#创建一个没有初始化的5*3矩阵：
#x = torch.empty(5, 3)
#print(x)

#创建一个随机初始化矩阵：
x = torch.rand(5, 3)
#print(x)

#构造一个填满0且数据类型为long的矩阵:
x = torch.zeros(5, 3, dtype=torch.long)
#print(x)

#直接从数据构造张量：
x = torch.tensor([5.5, 3])
#print(x)

#或者根据已有的tensor建立新的tensor。除非用户提供新的值，否则这些方法将重用输入张量的属性，例如dtype等：
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
#print(x)

x = torch.randn_like(x, dtype=torch.float)    # 重载 dtype!
#print(x)                                      # 结果size一致

#获取张量的形状：
#print(x.size())

#运算
#一种运算有多种语法。在下面的示例中，我们将研究加法运算。
#加法：形式一
y = torch.rand(5, 3)
#print(x + y)

#加法：形式二
#print(torch.add(x, y))

#加法：给定一个输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
#print(result)

#加法：原位/原地操作(in-place）
# adds x to y
y.add_(x)
#print(y)

#也可以使用像标准的NumPy一样的各种索引操作：
#print(x[:, 1])

#改变形状：如果想改变形状，可以使用torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
#print(x.size(), y.size(), z.size())

#如果是仅包含一个元素的tensor，可以使用.item()来得到对应的python数值
x = torch.randn(1)
#print(x)
#print(x.item())

#桥接 NumPy
#将torch的Tensor转化为NumPy数组
a = torch.ones(5)
#print(a)

b = a.numpy()
#print(b)

#看NumPy数组是如何改变里面的值的：
a.add_(1)
#print(a)
#print(b)

#将NumPy数组转化为Torch张量
#看改变NumPy数组是如何自动改变Torch张量的：
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

#CUDA上的张量
#张量可以使用.to方法移动到任何设备(device）上：
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    #print(z)
    #print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype



























































































