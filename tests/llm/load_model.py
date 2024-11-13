from swift.llm import get_model_tokenizer
import argparse


def get_model_and_tokenizer(ms_model_id):
    try:
        import transformers
        print(f'Test model: {ms_model_id} with transformers version: {transformers.__version__}')
        model_ins, tokenizer = get_model_tokenizer(ms_model_id)
    except Exception:
        import traceback
        print(traceback.format_exc())
        raise


parser = argparse.ArgumentParser()
parser.add_argument(
    '--ms_model_id',
    type=str,
    required=True,
)
args = parser.parse_args()

get_model_and_tokenizer(args.ms_model_id)

