from __future__ import division

#创建一个txt文件,文件名为mytxtfile,并向文件写入msg
def txt_creat(name,msg):
    print('俺进来这个方法啦！')
    desktop_path = "C:\\Users\\Administrator\\Desktop\\"
    full_path = desktop_path + name + '.txt'
    file = open(full_path,'w')
    file.write(msg)




if __name__ == '__main__':
    txt_creat('mytxtfile', '我在测试文件')