#Docker 镜像加速

#国内从 DockerHub 拉取镜像有时会遇到困难，此时可以配置镜像加速器。
#Docker 官方和国内很多云服务商都提供了国内加速器服务，例如：
#1.网易：https://hub-mirror.c.163.com/
#2.阿里云：https://<你的ID>.mirror.aliyuncs.com
#3.七牛云加速器：https://reg-mirror.qiniu.com

#当配置某一个加速器地址之后，若发现拉取不到镜像，请切换到另一个加速器地址。
#国内各大云服务商均提供了 Docker 镜像加速服务，建议根据运行 Docker 的云平台选择对应的镜像加速服务。
#阿里云镜像获取地址：https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors，登陆后，左侧菜单选中镜像加速器就可以看到你的专属地址了：
#之前还有 Docker 官方加速器 https://registry.docker-cn.com ，现在好像已经不能使用了，我们可以多添加几个国内的镜像，如果有不能使用的，会切换到可以使用个的镜像来拉取。

