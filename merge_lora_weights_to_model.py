import os.path
import shutil

from modelscope import snapshot_download

from swift import Swift
from swift.llm import InferArguments, get_model_tokenizer
from swift.utils import parse_args


def convert_ckpt(args):
    assert args.model_id_or_path is not None and args.ckpt_dir is not None
    if not os.path.exists(args.model_id_or_path):
        args.model_id_or_path = snapshot_download(args.model_id_or_path)

    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, device_map='cpu')

    # ### Preparing LoRA
    model = Swift.from_pretrained(model, args.ckpt_dir, inference_mode=True)
    Swift.merge_and_unload(model)
    model.model.save_pretrained(args.ckpt_dir + '-merged')
    tokenizer.save_pretrained(args.ckpt_dir + '-merged')
    for fname in os.listdir(args.ckpt_dir):
        if fname in {'generation_config.json'}:
            src_path = os.path.join(args.ckpt_dir, fname)
            tgt_path = os.path.join(args.ckpt_dir + '-merged', fname)
            shutil.copy(src_path, tgt_path)
    print(f'model saved to : {args.ckpt_dir + "-merged"}')


if __name__ == '__main__':
    args, remaining_argv = parse_args(InferArguments, None)
    args.init_argument()
    convert_ckpt(args)
