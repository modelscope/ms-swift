import orjson
import jsonlines
import subprocess
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count


def get_files(input_dir, suffix):
    result = subprocess.run(
        [
            'find', str(input_dir),
            '-maxdepth', '1',
            '-type', 'f',
            '-name', f'*{suffix}'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )
    return result.stdout.splitlines()


def get_paired_files(input_dir):
    json_files = get_files(input_dir, '.json')
    jpg_files = get_files(input_dir, '.jpg')

    json_stems = {Path(p).stem for p in json_files}
    jpg_stems = {Path(p).stem for p in jpg_files}

    paired_stems = json_stems & jpg_stems

    return [input_dir / f'{stem}.json' for stem in paired_stems]


def process_subset(args):
    parent_dir, subset = args
    input_dir = parent_dir / subset
    output_file = input_dir.with_suffix('.jsonl')

    json_files = get_paired_files(input_dir)

    with jsonlines.open(output_file, mode='w') as writer:
        for json_file in json_files:
            try:
                with open(json_file, 'rb') as f:
                    data = orjson.loads(f.read())
                    if isinstance(data, list):
                        writer.write_all(data)
                    elif isinstance(data, dict):
                        writer.write(data)
            except Exception:
                pass

    print(f'{subset}: {len(json_files)}')


def main():
    parent_dir = Path('/nas_train/app.e0016372/datasets/LLaVA-OneVision-1.5-Mid-Training-85M')

    subsets = [
        'coyo',
        'datacomp1b',
        'imagenet',
        'laioncn',
        'mint',
    ]

    num_workers = min(len(subsets), cpu_count())
    tasks = [(parent_dir, subset) for subset in subsets]

    with Pool(processes=num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(process_subset, tasks),
            total=len(tasks),
            desc='Processing subsets'
        ))

    print('\n=== 完成 ===')


if __name__ == '__main__':
    main()
