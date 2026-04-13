import os
import jsonlines
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count


def resolve_image_path(data, img_root, field):
    if field not in data:
        return data

    value = data[field]

    if isinstance(value, str):
        p = Path(value)
        if not p.is_absolute():
            p = img_root / p
        data[field] = str(p)
    elif isinstance(value, list):
        new_value = []
        for v in value:
            if isinstance(v, str):
                p = Path(v)
                if not p.is_absolute():
                    p = img_root / p
                new_value.append(str(p))
            else:
                new_value.append(v)
        data[field] = new_value

    return data


def process_file(jsonl_path, img_root, field):
    out_path = jsonl_path.with_suffix('.tmp')

    with jsonlines.open(jsonl_path, 'r') as reader, \
         jsonlines.open(out_path, 'w') as writer:

        for data in reader:
            if not isinstance(data, dict):
                continue

            data = resolve_image_path(data, img_root, field)
            writer.write(data)

    os.replace(out_path, jsonl_path)


def main(args):
    jsonl_dir = Path(args.jsonl_dir)
    img_root = Path(args.img_root)

    files = sorted(jsonl_dir.glob('**/*.jsonl'))
    print(f'Found {len(files)} jsonl files')

    if len(files) == 0:
        return

    worker = partial(
        process_file,
        img_root=img_root,
        field=args.field
    )

    with Pool(min(cpu_count(), len(files))) as pool:
        list(tqdm(
            pool.imap_unordered(worker, files),
            total=len(files),
            desc='Processing',
        ))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_dir', type=str, required=True)
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--field', type=str, default='images')

    args = parser.parse_args()

    main(args)
