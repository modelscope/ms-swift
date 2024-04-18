# 安装和使用

## Wheel包安装

可以使用pip进行安装：

```shell
# 全量能力
pip install 'ms-swift[all]' -U
# 仅使用LLM
pip install 'ms-swift[llm]' -U
# 仅使用AIGC
pip install 'ms-swift[aigc]' -U
# 仅使用adapters
pip install ms-swift -U
```

## 源代码安装

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[all]'
```

## Notebook环境

Swift支持训练的绝大多数模型都可以在`A10`显卡上使用，用户可以使用ModelScope官方提供的免费显卡资源：

1. 进入[ModelScope](https://www.modelscope.cn)官方网站并登录
2. 点击左侧的`我的Notebook`并开启一个免费GPU实例
3. 愉快地薅A10显卡羊毛

## Build文档

Swift支持完整的API Doc文档，在swift根目录下执行：

```shell
make docs
```

等待执行完成后，查看`docs/build/html/index.html`即可。
