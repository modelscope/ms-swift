import re
import shutil
import filetype
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from multiprocessing import Pool, cpu_count


DATA_ROOT = Path(
    # '/nas_train/app.e0031982/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data'
    '/nas_train/app.e0031982/datasets/FineVision'
)
OUTPUT_ROOT = Path(
    # '/nas_user/app.e0016372/datasets/LLaVA-OneVision-1.5-Instruct-Data'
    '/nas_user/app.e0016372/datasets/FineVision'
)

ERROR_LOG = OUTPUT_ROOT / 'error.log'
IMAGE_ROOT = OUTPUT_ROOT / 'images'
JSONL_ROOT = OUTPUT_ROOT / 'jsonl'


def normalize_images(images):
    if isinstance(images, (list, np.ndarray)):
        return images
    elif isinstance(images, dict):
        return [images]
    else:
        return []


def validate_messages(messages):
    if not messages:
        raise ValueError('messages empty after filtering system messages')
    if messages[0]['role'] != 'user':
        raise ValueError('conversation must start with user')
    for i in range(1, len(messages)):
        if messages[i - 1]['role'] == messages[i]['role']:
            raise ValueError(
                f'roles not alternating at index {i}: '
                f"{messages[i - 1]['role']} -> {messages[i]['role']}"
            )


def ensure_image_prefix(messages, num_images):
    if num_images <= 0:
        return
    content = messages[0]['content']
    for p in (r'^\s*(?:<image>\s*)+', r'(?:\s*<image>)+\s*$'):
        content = re.sub(p, '', content)
    prefix = '\n'.join(['<image>'] * num_images)
    messages[0]['content'] = f'{prefix}\n{content}' if content else prefix


def normalize_messages(raw_messages, num_images=0):
    messages = []
    for idx, item in enumerate(raw_messages):
        if role := item.get('from'):
            if role == 'system':
                continue
            role_map = {'human': 'user', 'gpt': 'assistant'}
            if role not in role_map:
                raise ValueError(f'message[{idx}] invalid from: {role}')
            role = role_map[role]
            content = item.get('value')
        elif role := item.get('role'):
            if role == 'system':
                continue
            if role not in {'user', 'assistant'}:
                raise ValueError(f'message[{idx}] invalid role: {role}')
            content = item.get('content')
        else:
            assert item.get('user')
            messages.append({'role': 'user', 'content': item['user']})
            role, content = 'assistant', item['assistant']

        if not content:
            raise ValueError(f'message[{idx}] empty content')

        messages.append({'role': role, 'content': content})

    validate_messages(messages)
    ensure_image_prefix(messages, num_images)
    return messages


def save_image(image_data, image_dir, sample_name):
    image_bytes = image_data['bytes']
    suffix = filetype.guess_extension(image_bytes) or 'png'
    image_path = image_dir / f'{sample_name}.{suffix}'
    image_path.write_bytes(image_bytes)
    return str(image_path)


def process_sample(row, dataset_name, parquet_stem, idx, image_dir):
    images = normalize_images(row.images)
    record = {
        'id': f'{dataset_name}/{parquet_stem}_{idx}',
        'messages': normalize_messages(row.messages, num_images=len(images)),
    }
    if len(images) > 0:
        record['images'] = [
            save_image(image, image_dir, f'{parquet_stem}_{idx}_{i}')
            for i, image in enumerate(images)
        ]
    return record


def process_parquet(parquet_path):
    dataset_name = parquet_path.parent.name
    image_dir = IMAGE_ROOT / dataset_name
    jsonl_dir = JSONL_ROOT / dataset_name

    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    output_file = jsonl_dir / f'{parquet_path.stem}.jsonl'

    try:
        df = pd.read_parquet(parquet_path, columns=['texts', 'images'])
        # df = pd.read_parquet(parquet_path, columns=['conversations', 'image'])
        df.columns = ['messages', 'images']

        with jsonlines.open(output_file, mode='w') as writer:
            for idx, row in enumerate(df.itertuples(index=False)):
                try:
                    record = process_sample(row, dataset_name, parquet_path.stem, idx, image_dir)
                    writer.write(record)
                except Exception as e:
                    logger.error(f'{parquet_path} | row={idx} | {repr(e)}')
    except Exception as e:
        logger.error(f'{parquet_path} | parquet_error | {repr(e)}')


def main():
    shutil.rmtree(IMAGE_ROOT, ignore_errors=True)
    shutil.rmtree(JSONL_ROOT, ignore_errors=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(ERROR_LOG, level='ERROR', enqueue=True)

    parquet_files = list(DATA_ROOT.glob('*/*.parquet'))
    # parquet_files = parquet_files[:2]

    print(len(parquet_files))
    if not parquet_files: return

    n_workers = min(cpu_count(), len(parquet_files))
    with Pool(processes=n_workers, maxtasksperchild=10) as pool:
        list(tqdm(
            pool.imap_unordered(process_parquet, parquet_files), 
            total=len(parquet_files)
        ))


if __name__ == '__main__':
    main()
