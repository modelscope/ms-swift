"""Download the public, pinned typed-decisions source files."""
import json
from pathlib import Path
import shutil
from huggingface_hub import HfApi, hf_hub_download

root = Path(__file__).resolve().parent
revision = 'd0e2f0c42fef86cc15d1688d25a19f5ba7c85b18'
repo = 'LocalLLaMA/typed-decisions'
(root / 'raw').mkdir(exist_ok=True)
(root / 'sources').mkdir(exist_ok=True)
info = HfApi().dataset_info(repo, revision=revision)
assert info.sha == revision
(root / 'sources/typed-meta.json').write_text(json.dumps({'sha': info.sha}))
for split in ['train', 'test']:
    source = hf_hub_download(repo, 'all/' + split + '-00000-of-00001.parquet',
                             repo_type='dataset', revision=revision)
    shutil.copy2(source, root / 'raw' / (split + '.parquet'))
