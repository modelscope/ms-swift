# SWIFT安装

## Wheel包安装

可以使用pip进行安装：

```shell
pip install 'ms-swift'
# 使用评测
pip install 'ms-swift[eval]' -U
# 使用序列并行
pip install 'ms-swift[seq_parallel]' -U
# 全能力
pip install 'ms-swift[all]' -U
```

## 源代码安装

```shell
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# 全能力
# pip install -e '.[all]'
```

## 旧版本

SWIFT在3.0版本开始进行了不兼容式重构，如果需要使用2.x旧版本，请执行如下命令进行安装：
```shell
pip install ms-swift==2.*
```

## 镜像

镜像可以查看[这里](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F)

你也可以使用[install_all.sh](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh)进行安装

## Notebook环境

Swift支持训练的绝大多数模型都可以在`A10`显卡上使用，用户可以使用ModelScope官方提供的免费显卡资源：

1. 进入[ModelScope](https://www.modelscope.cn)官方网站并登录
2. 点击左侧的`我的Notebook`并开启一个免费GPU实例
3. 愉快地薅A10显卡羊毛
