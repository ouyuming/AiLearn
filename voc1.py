import os
import cv2


# 遍历指定目录，显示目录下的所有文件名
def CropImage4File(filepath, destpath):
    pathDir = os.listdir(filepath)  # 列出文件路径中的所有路径或文件
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        dest = os.path.join(destpath, allDir)
        if os.path.isfile(child):
            image = cv2.imread(child)
        sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
        sz1 = sp[0]  # 图像的高度（行 范围）
        sz2 = sp[1]  # 图像的宽度（列 范围）
        # sz3 = sp[2]                #像素值由【RGB】三原色组成

        # 你想对文件的操作
        a = int(sz1 / 2 - 64)  # x start
        b = int(sz1 / 2 + 64)  # x end
        c = int(sz2 / 2 - 64)  # y start
        d = int(sz2 / 2 + 64)  # y end
        cropImg = image[a:b, c:d]  # 裁剪图像
        cv2.imwrite(dest, cropImg)  # 写入图像路径


if __name__ == '__main__':
    filepath = r'C:/Users/ouyuming/Desktop/voc_tool/VOCdevkit/VOC2012/JPEGImages/'  # 源图像
    destpath = r'C:/Users/ouyuming/Desktop/exercise/'  # resized images saved here
    CropImage4File(filepath, destpath)