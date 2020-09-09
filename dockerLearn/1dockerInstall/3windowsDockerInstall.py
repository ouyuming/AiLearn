#Windows Docker 安装

#Win10 系统
#现在 Docker 有专门的 Win10 专业版系统的安装包，需要开启 Hyper-V。

#开启 Hyper-V
#本人电脑缺失Hyper-V
#链接：https://docs.microsoft.com/zh-cn/virtualization/hyper-v-on-windows/quick-start/enable-hyper-v
#升级为win10专业版密钥: https://blog.csdn.net/qq_36938933/article/details/79429481

#程序和功能
#启用或关闭Windows功能
#选中Hyper-V


#1、安装 Toolbox
#最新版 Toolbox 下载地址： 访问 https://www.docker.com/get-started，注册一个账号，然后登录。
#点击 Get started with Docker Desktop，并下载 Windows 的版本，如果你还没有登录，会要求注册登录：

#2、运行安装文件
#双击下载的 Docker for Windows Installer 安装文件，一路 Next，点击 Finish 完成安装。
#安装完成后，Docker 会自动启动。通知栏上会出现个小鲸鱼的图标，这表示 Docker 正在运行。
#桌边也会出现三个图标，入下图所示：
#我们可以在命令行执行 docker version 来查看版本号，docker run hello-world 来载入测试镜像测试。
#如果没启动，你可以在 Windows 搜索 Docker 来启动：
#启动后，也可以在通知栏上看到小鲸鱼图标：

#镜像加速
#Windows 10
#对于使用 Windows 10 的系统，在系统右下角托盘 Docker 图标内右键菜单选择 Settings，打开配置窗口后左侧导航菜单选择 Daemon。
#在 Registrymirrors 一栏中填写加速器地址 https://registry.docker-cn.com ，之后点击 Apply 保存后 Docker 就会重启并应用配置的镜像地址了。






































