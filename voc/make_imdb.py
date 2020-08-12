import os
from PIL import Image
import pickle
import sys
from torchvision import datasets, transforms
import torch

sys.setrecursionlimit(10000000)

data_path = 'C:/Users/ouyuming/Desktop/pyProjects/allFile/exercise/'
phase = 'test'

#totalpath = '{}/{}'.format(data_path, phase)
totalpath = data_path
sub_dirs = os.listdir(totalpath)
imgs = []
lbls = []
filelists = []
print(type(filelists))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#for na in sub_dirs:
for i,na in enumerate(sub_dirs):
    path0 = '{}/{}'.format(totalpath, na)
    imagename = os.listdir(path0)
    for tu in imagename:
        imagenames = "{}/{}".format(path0, tu)
        image = Image.open(imagenames)

        # un=transforms.ToPILImage()
        image = transform(image)
        #lbl = torch.tensor(int(na))
        lbl = torch.tensor(int(i))
        

        filelists.append([image, lbl])

with open('{}/{}.{}'.format(data_path, phase, 'imdb'), "wb") as f:
    pickle.dump(filelists, f)  #