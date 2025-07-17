
from ..dataset import EncodePreprocessor, IterablePackingDataset, LazyLLMDataset, PackingDataset, load_dataset

def encode_dataset(args, template, dataset):
    if args.packing:
        dataset = PackingDataset(
            template,
            dataset,
            num_proc=args.dataset_num_proc,
            strict=args.strict,
            load_from_cache_file=args.load_from_cache_file)
    else:
        preprocessor = EncodePreprocessor(template=template)
        dataset = preprocessor(
            dataset,
            num_proc=args.dataset_num_proc,
            load_from_cache_file=args.load_from_cache_file,
            strict=args.strict)
    return dataset

def export_to_bin_dataset(args):
    _, processor = args.get_model_processor(load_model=False)
    template = args.get_template(processor)
    from ..train import SwiftSft
    train_dataset, val_dataset = SwiftSft._get_dataset(args)

    for dataset_type, dataset in [('train', train_dataset), ('val', val_dataset)]:
        dataset = encode_dataset(args, template, dataset)
