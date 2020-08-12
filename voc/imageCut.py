from __future__ import division

import concurrent.futures
import os
from PIL import Image
import xml.dom.minidom
import numpy as np
import math


# 未处理图片的存放路径
JpgPath = r'C:/Users/ouyuming/Desktop/pyProjects/allFile/VOC2007/JPEGImages/'

# 处理后图片的存放路径
ProcessedPath = r'C:/Users/ouyuming/Desktop/pyProjects/allFile/exercise/'
#xml的存放路径
AnnoPath = r'C:/Users/ouyuming/Desktop/pyProjects/allFile/VOC2007/Annotations/'

#创建一个txt文件,文件名为mytxtfile,并向文件写入msg
def txt_creat(name,path):
    print('俺进来这个方法啦！')
    #路径问题多注意！
    desktop_path = "C:/Users/ouyuming/Desktop/pyProjects/allFile/"
    full_path = desktop_path + 'voc.txt'
    file = open(full_path,'a+') #新的内容将会被写入到已有内容之后
    msg = path + ":" + name + "\n"
    print(msg)
    file.write(msg)




def get_image(file_name):
    #获取图片
    image_name, ext = os.path.splitext(file_name) #分割路径中的文件名与拓展名
    imgfile = JpgPath + file_name #拼接图片的路径，找到指定路径
    print('正在处理图像:'+ imgfile)
    xmlfile = AnnoPath + image_name + '.xml' #找到待拼接图片的xml存储路径
    print('正在处理XML:' + xmlfile)
    DomTree = xml.dom.minidom.parse(xmlfile) #将所有元素保存在树结构里
    annotation = DomTree.documentElement
    objectlist = annotation.getElementsByTagName('object')

    #切割图片(利用for循环)
    i = 0
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data #获取类型名称
        savepath = ProcessedPath + objectname #分类保存路径
        if not os.path.exists(savepath): #路径不存在则创建路径
            os.makedirs(savepath)
        bndboxs = objects.getElementsByTagName('bndbox') #通过bndbox标签获取切割四个坐标
        x1_list = bndboxs[0].getElementsByTagName('xmin')
        x1_list = bndboxs[0].getElementsByTagName('xmin')
        x1 = math.ceil(float(x1_list[0].childNodes[0].data)) #获取x轴最小坐标
        y1_list = bndboxs[0].getElementsByTagName('ymin')
        y1 = math.ceil(float(y1_list[0].childNodes[0].data)) #获取y轴最小坐标
        x2_list = bndboxs[0].getElementsByTagName('xmax')
        x2 = math.ceil(float(x2_list[0].childNodes[0].data)) #获取x轴最大坐标
        y2_list = bndboxs[0].getElementsByTagName('ymax')
        y2 = math.ceil(float(y2_list[0].childNodes[0].data)) #获取y轴最大坐标
        crop_box = np.array([x1, y1, x2, y2]) #获取坐标数组
        img = Image.open(imgfile) #打开图片
        cropedimg = img.crop(crop_box) #根据数组坐标裁剪图片(pillow模块Image.crop()函数切割图片)
        i += 1 #每张图片里面，对象的个数
        print(str(i))
        allPath = savepath + '/' + image_name + '_' + str(i) + '.jpg'
        cropedimg.save(allPath) #保存在给定文件名下的该图像
        #C:/Users/Administrator/Desktop/exercise/aeroplane/000067_1.jpg
        #将图片的路径和类型保存到txt;image_name是图片类型;
        txt_creat(objectname, allPath)




if __name__ == '__main__':
    image_list = os.listdir(JpgPath)  # 拿到JpgPath路径下的所有图片
    #需要理解透这句代码的执行原理(这个多线程貌似控制不太好)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(get_image, image_list)
    print('图片获取完成 。。。！')



