import os
from typing import Dict, List, Optional
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as HfDataset


def _load_dataset_path(
    dataset_path: str,
    *,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
    strict: bool = False,
    streaming: bool = False,
    columns: Optional[Dict[str, str]] = None,
    remove_unused_columns: bool = True,
    exclude_fields: Optional[List[str]] = None,
) -> HfDataset:
    exclude_fields = ['metadata']
    ext = os.path.splitext(dataset_path)[1].lstrip('.')
    
    # 处理YAML配置文件
    import yaml
    import glob 
    if ext.lower() in ['yaml', 'yml']:
        with open(dataset_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 收集所有JSONL文件路径
        jsonl_files = []
        for dataset_info in config['Datasets'].values():
            meta_dir = dataset_info['MetaFiles']
            jsonl_files.extend(glob.glob(os.path.join(meta_dir, '*.jsonl')))
        
        if not jsonl_files:
            raise ValueError(f"No JSONL files found in MetaFiles directories specified in {dataset_path}")
        
        # 加载所有JSONL文件作为数据集
        kwargs = {'split': 'train', 'streaming': streaming, 'num_proc': num_proc}
        
        # 如果有需要排除的字段，使用自定义加载器
        if exclude_fields:
            dataset = _load_jsonl_with_excluded_fields(jsonl_files, exclude_fields, **kwargs)
        else:
            dataset = hf_load_dataset('json', data_files=jsonl_files, **kwargs)
    
    # 原始文件处理逻辑
    else:
        file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext
        kwargs = {'split': 'train', 'streaming': streaming, 'num_proc': num_proc}
        if file_type == 'csv':
            kwargs['na_filter'] = False
        
        # 对单个JSONL文件使用自定义加载器
        if file_type == 'json' and exclude_fields:
            dataset = _load_jsonl_with_excluded_fields([dataset_path], exclude_fields, **kwargs)
        else:
            dataset = hf_load_dataset(file_type, data_files=dataset_path, **kwargs)
    
    return dataset


def _load_jsonl_with_excluded_fields(
        jsonl_files: List[str],
        exclude_fields: List[str],
        **kwargs
    ) -> HfDataset:
    """
    自定义JSONL加载器，在解析时直接排除指定字段
    """
    # 创建自定义生成器函数
    import json

    def jsonl_generator():
        for file_path in jsonl_files:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # 排除指定字段
                        for field in exclude_fields:
                            if field in data:
                                del data[field]
                        yield data
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误在文件 {file_path}: {e}")
                        continue
                    except Exception as e:
                        print(f"处理行时出错: {e}")
                        continue
    
    # 使用生成器创建数据集
    return HfDataset.from_generator(
        jsonl_generator,
        **kwargs
    )


dataset = _load_dataset_path("/workspace/home/chenjiali/cjl-test-1/cjl-pck/ms-swift-cjl/test.yaml")

print("111")