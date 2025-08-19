# Copyright (c) Alibaba, Inc. and its affiliates.
# !/usr/bin/env python
import os
from setuptools import find_packages, setup
from typing import List


def readme():
    """
    读取根目录下的 README.md 文件内容并返回。

    Returns:
        str: README.md 的完整文本内容。
    """
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'swift/version.py'


def get_version():
    """
    读取版本文件并返回 __version__。

    Returns:
        str: 版本号字符串（例如 '3.7.2'）。
    """
    with open(version_file, 'r', encoding='utf-8') as f:
        # 编译并执行文件内容，将其中定义的变量（如 __version__）注入当前作用域
        exec(compile(f.read(), version_file, 'exec'))
    # 从当前局部变量中获取 __version__ 并返回
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    # 导入正则表达式模块，用于解析依赖行中的版本运算符
    import re
    # 导入 sys 模块，用于判断 Python 版本等信息
    import sys
    # 从 os.path 导入 exists，用于判断文件是否存在
    from os.path import exists
    # 记录入口 requirements 文件的路径，供后续解析使用
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        # 判断是否为包含其他 requirements 文件的引用（-r path）
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            # 从当前行中解析被引用的目标文件路径
            target = line.split(' ')[1]
            # 获取当前入口文件的目录，便于拼接相对路径
            relative_base = os.path.dirname(fname)
            # 组合得到被引用文件的绝对路径
            absolute_target = os.path.join(relative_base, target)
            # 逐条产出被引用文件中的依赖解析结果
            for info in parse_require_file(absolute_target):
                yield info
        else:
            # 初始化当前行的解析结果字典（至少保留原始行内容）
            info = {'line': line}
            # 处理可编辑安装（-e git+...#egg=pkg）
            if line.startswith('-e '):
                # 从 #egg= 片段中提取包名
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                # 构造用于匹配版本关系运算符的正则，例如 >=、==、>
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                # 最多按照一次版本分隔符将当前行切片
                parts = re.split(pat, line, maxsplit=1)
                # 去除切片结果中每个元素的首尾空白
                parts = [p.strip() for p in parts]
                # 将包名部分写入解析结果
                info['package'] = parts[0]
                # 若包含版本信息则进一步解析
                if len(parts) > 1:
                    # 解包出运算符和后续字符串（可能包含平台限定）
                    op, rest = parts[1:]
                    # 若包含平台依赖分隔符 ;，则拆出版本与平台条件
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        # 拆分版本与平台条件，并去除首尾空白
                        version, platform_deps = map(str.strip, rest.split(';'))
                        # 记录平台条件，供后续拼接
                        info['platform_deps'] = platform_deps
                    else:
                        # 不包含平台条件时，rest 全部作为版本字符串
                        version = rest  # NOQA
                    # 记录版本关系（运算符与版本字符串二元组）
                    info['version'] = (op, version)
            # 返回当前行的解析结果
            yield info

    def parse_require_file(fpath):
        """
        从指定的 requirements 文件中逐行解析依赖项。

        Args:
            fpath (str): requirements 文件路径。

        Yields:
            dict: 每一条依赖对应的解析结果，或包含 dependency_links 的字典。
        """
        # 以只读方式打开指定的 requirements 文件，使用 UTF-8 编码
        with open(fpath, 'r', encoding='utf-8') as f:
            # 遍历文件中的每一行
            for line in f.readlines():
                # 去除当前行首尾空白字符
                line = line.strip()
                # 跳过以 http 开头的行（通常是直接的链接依赖）
                if line.startswith('http'):
                    print('skip http requirements %s' % line)
                    continue
                # 对非空、且不以注释 # 或选项 -- 开头的行进行解析
                if line and not line.startswith('#') and not line.startswith('--'):
                    # 将该行交给 parse_line 解析并逐条产出
                    for info in parse_line(line):
                        yield info
                # 处理形如 --find-links 的选项行，从中提取依赖链接
                elif line and line.startswith('--find-links'):
                    # 将选项行按空白分割为片段
                    eles = line.split()
                    # 遍历每个片段，查找包含 http 的链接
                    for e in eles:
                        e = e.strip()
                        # 将包含 http 的片段作为 dependency_links 产出
                        if 'http' in e:
                            info = dict(dependency_links=e)
                            yield info

    def gen_packages_items():
        """
        根据解析结果生成 install_requires 列表和依赖链接集合。

        Returns:
            Tuple[List[str], List[str]]: 安装依赖项列表和依赖链接列表。
        """
        # 初始化安装依赖项列表
        items = []
        # 初始化依赖链接列表（如 --find-links 中的链接）
        deps_link = []
        # 若入口 requirements 文件存在，则进行解析
        if exists(require_fpath):
            # 遍历入口 requirements 文件中解析得到的每条信息
            for info in parse_require_file(require_fpath):
                # 若该条信息不是依赖链接，则构造标准依赖字符串
                if 'dependency_links' not in info:
                    # 以包名为基础
                    parts = [info['package']]
                    # 按需拼接版本关系（运算符与版本）
                    if with_version and 'version' in info:
                        parts.extend(info['version'])
                    # Python 3.4 下 package_deps 存在问题，跳过平台依赖拼接
                    if not sys.version.startswith('3.4'):
                        # apparently package_deps are broken in 3.4
                        # 读取平台依赖条件并按规范拼接
                        platform_deps = info.get('platform_deps')
                        if platform_deps is not None:
                            parts.append(';' + platform_deps)
                    # 将各部分合并为最终依赖字符串
                    item = ''.join(parts)
                    # 加入安装依赖项列表
                    items.append(item)
                else:
                    # 记录依赖链接，供 setup(...) 的 dependency_links 使用
                    deps_link.append(info['dependency_links'])
        # 返回安装依赖项与依赖链接
        return items, deps_link

    # 执行内部生成函数，返回依赖项列表与依赖链接
    return gen_packages_items()


"""
当作为脚本直接执行时（例如 pip install -e . 触发），运行下面的安装与打包配置。
"""
if __name__ == '__main__':
    # 解析入口 requirements.txt，返回安装依赖列表（install_requires）与依赖链接（dependency_links）
    install_requires, deps_link = parse_requirements('requirements.txt')
    # 初始化可选依赖的集合（键为功能组名，值为依赖列表）
    extra_requires = {}
    # 初始化一个列表，用于聚合所有可选依赖，形成 extras_require['all']
    all_requires = []
    # 解析评测相关依赖，放入 extras_require['eval']，忽略返回的依赖链接
    extra_requires['eval'], _ = parse_requirements('requirements/eval.txt')
    # 解析 swanlab 相关依赖，放入 extras_require['swanlab']，忽略返回的依赖链接
    extra_requires['swanlab'], _ = parse_requirements('requirements/swanlab.txt')
    # 将基础依赖加入 all_requires，便于组合形成一个 "all" 选项
    all_requires.extend(install_requires)
    # 将 eval 依赖加入 all_requires
    all_requires.extend(extra_requires['eval'])
    # 将 swanlab 依赖加入 all_requires
    all_requires.extend(extra_requires['swanlab'])
    # 将聚合后的依赖集合注册为 extras_require['all']
    extra_requires['all'] = all_requires

    # 调用 setuptools.setup(...) 执行安装与元数据配置
    setup(
        # 包发布名称（distribution name），与 import 包名不同；此处发布名为 ms_swift
        name='ms_swift',
        # 版本号，从 swift/version.py 中动态读取
        version=get_version(),
        # 项目简短描述，将显示在 PyPI 等处
        description='Swift: Scalable lightWeight Infrastructure for Fine-Tuning',
        # 项目长描述，通常用于 PyPI 项目页详情
        long_description=readme(),
        # 指定长描述的内容类型为 Markdown
        long_description_content_type='text/markdown',
        # 作者信息
        author='DAMO ModelScope teams',
        # 作者联系邮箱
        author_email='contact@modelscope.cn',
        # 关键词，便于检索
        keywords='python, petl, efficient tuners',
        # 项目主页链接
        url='https://github.com/modelscope/swift',
        # 自动发现并包含包（排除 configs 与 demo 目录）
        packages=find_packages(exclude=('configs', 'demo')),
        # 打包时包含包数据（结合 MANIFEST.in / package_data 生效）
        include_package_data=True,
        # 显式指定需要包含到包中的数据文件类型
        package_data={
            # 对所有包，包含 .h / .cpp / .cu 源文件（部分扩展或编译相关文件）
            '': ['*.h', '*.cpp', '*.cu'],
        },
        # PyPI 分类器（项目成熟度、许可证、平台与 Python 版本等元数据）
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
        ],
        # 许可协议，Apache License 2.0
        license='Apache License 2.0',
        # 测试依赖（开发/测试环境可用），从 requirements/tests.txt 解析
        tests_require=parse_requirements('requirements/tests.txt'),
        # 安装所需的基础依赖
        install_requires=install_requires,
        # 额外可选依赖分组（如 eval、swanlab、all）
        extras_require=extra_requires,
        # 安装后生成的命令行入口映射
        entry_points={
            'console_scripts': ['swift=swift.cli.main:cli_main', 'megatron=swift.cli._megatron.main:cli_main']
        },
        # 依赖链接（例如 --find-links 指定的自定义索引/镜像）
        dependency_links=deps_link,
        # 指定包内容在压缩环境中是否可用；False 通常表示解压到文件系统使用
        zip_safe=False)
