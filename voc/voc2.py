"""
处理数据集 和 标签数据集的代码：（主要是对原始数据集裁剪）
    处理方式：分别处理
    注意修改 输入 输出目录 和 生成的文件名
    output_dir = "./label_temp"
    input_dir = "./label"
"""
import cv2
import os
import sys
import time
from xml.dom.minidom import parse
import xml.dom.minidom



#获取图片
def get_img(input_dir):
    img_paths = []
    for (path,dirname,filenames) in os.walk(input_dir):
        for filename in filenames:
            img_paths.append(path+'/'+filename)
    print("img_paths:",img_paths)
    return img_paths

#裁剪图片
def cut_img(img_paths,output_dir):
    image_list = os.listdir(img_paths)
    #根据xml里面的坐标进行裁剪
    scale = len(img_paths)
    for i,img_path in enumerate(img_paths): #遍历每一张图

        #image_pre, ext = os.path.splitext(img_path)
        # 拿到xml的数据集中的坐标
        #xmlFilePath = AnnoPath + image_pre + '.xml'  # 找到待拼接图片的xml存储路径
        #DomTree = xml.dom.minidom.parse(xmlFilePath)  # 将所有元素保存在树结构里
        #annotation = DomTree.documentElement  # 在这里出问题了！

        a = "#"* int(i/1000)
        b = "."*(int(scale/1000)-int(i/1000))
        c = (i/scale)*100
        time.sleep(0.2)
        print('正在处理图像： %s' % img_path.split('/')[-1])
        img = cv2.imread(img_path)
        weight = img.shape[1]
        if weight>1600:                         # 正常发票
            cropImg = img[50:200, 700:1500]    # 裁剪【y1,y2：x1,x2】
            #cropImg = cv2.resize(cropImg, None, fx=0.5, fy=0.5,
                                 #interpolation=cv2.INTER_CUBIC) #缩小图像
            cv2.imwrite(output_dir + '/' + img_path.split('/')[-1], cropImg)
        else:                                        # 卷帘发票
            cropImg_01 = img[30:150, 50:600]
            cv2.imwrite(output_dir + '/'+img_path.split('/')[-1], cropImg_01)
        print('{:^3.3f}%[{}>>{}]'.format(c,a,b))

        #将裁剪后的图片存入目标文件夹


if __name__ == '__main__':
    output_dir = r'C:/Users/ouyuming/Desktop/exercise2/' # 保存截取的图像目录
    input_dir = r'C:/Users/ouyuming/Desktop/voc_tool/VOCdevkit/VOC2012/JPEGImages/' # 读取图片目录表
    AnnoPath = r'C:/Users/ouyuming/Desktop/voc_tool/VOCdevkit/VOC2012/Annotations/' # xml的存放路径
    img_paths = get_img(input_dir)
    print('图片获取完成 。。。！')
    cut_img(img_paths,output_dir)


