import json
import numpy as np

from swift.utils import append_to_jsonl, check_json_format


def test_check_json_format_converts_numpy_scalars():
    result = check_json_format({
        'accuracy': np.float32(0.75),
        'count': np.int64(3),
        'nested': [np.bool_(True)],
    })

    assert result == {'accuracy': 0.75, 'count': 3, 'nested': [True]}


def test_append_to_jsonl_converts_numpy_scalars(tmp_path):
    output_path = tmp_path / 'result.jsonl'

    append_to_jsonl(
        str(output_path),
        {
            'accuracy': np.float32(0.75),
            'count': np.int64(3),
            'nested': [np.bool_(True)],
        },
    )

    assert json.loads(output_path.read_text()) == {
        'accuracy': 0.75,
        'count': 3,
        'nested': [True],
    }
