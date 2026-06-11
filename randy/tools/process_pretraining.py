import re
import shutil
import hashlib
import filetype
import jsonlines
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from multiprocessing import Pool, cpu_count


DATA_ROOTS = [
    '/nas_train/app.e0031982/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/coyo',
    '/nas_train/app.e0031982/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/datacomp1b',
    '/nas_train/app.e0031982/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/imagenet',
    '/nas_train/app.e0031982/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/laioncn',
    '/nas_train/app.e0031982/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/mint',
    '/nas_train/app.e0031982/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/obelics',
    '/nas_train/app.e0016372/datasets/OmniScience/omniscience',
]

OUTPUT_ROOT = Path('/nas_user/app.e0016372/datasets/LLaVA-OneVision-1.5-Mid-Training-85M')
ERROR_LOG = OUTPUT_ROOT / 'error.log'


def compute_hash(x):
    return hashlib.md5(str(x).encode()).hexdigest()


def save_image(image_data, image_dir, sample_name):
    image_bytes = image_data['bytes']
    suffix = filetype.guess_extension(image_bytes) or 'png'
    image_path = image_dir / f'{sample_name}.{suffix}'
    image_path.write_bytes(image_bytes)
    return str(image_path)


def process_sample(row, parquet_path, idx, image_dir):
    if not row.caption or row.image is None:
        raise ValueError('empty caption or image')

    sample_name = compute_hash(f'{parquet_path}_{idx}')
    image_path = save_image(row.image, image_dir, sample_name)

    return {
        'messages': [
            {'role': 'user', 'content': '<image>'},
            {'role': 'assistant', 'content': row.caption}
        ],
        'images': [image_path]
    }


def process_parquet(args):
    parquet_path, dataset_name = args

    image_dir = OUTPUT_ROOT / dataset_name / 'images'
    jsonl_dir = OUTPUT_ROOT / dataset_name / 'jsonl'

    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = jsonl_dir / f'{compute_hash(parquet_path)}.jsonl'

    try:
        df = pd.read_parquet(parquet_path, columns=['caption', 'image'])

        with jsonlines.open(jsonl_path, 'w') as writer:
            for idx, row in enumerate(df.itertuples(index=False)):
                try:
                    record = process_sample(row, parquet_path, idx, image_dir)
                    writer.write(record)
                except Exception as e:
                    logger.error(f'{parquet_path} | row={idx} | {repr(e)}')
    except Exception as e:
        logger.error(f'{parquet_path} | parquet_error | {repr(e)}')


def main():
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(ERROR_LOG, level='ERROR', enqueue=True)

    tasks = []
    for root in DATA_ROOTS:
        root = Path(root)
        dataset_name = root.stem
        for p in root.glob('**/*.parquet'):
            tasks.append((p, dataset_name))

    print(len(tasks))
    if not tasks: return

    n_workers = min(cpu_count(), len(tasks))
    with Pool(processes=10, maxtasksperchild=10) as pool:
        list(tqdm(
            pool.imap_unordered(process_parquet, tasks),
            total=len(tasks)
        ))


if __name__ == '__main__':
    main()
