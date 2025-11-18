from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset


class CoundownTaskPreprocessor(ResponsePreprocessor):
    def preprocess(self, row):
        numbers = row['nums']
        target = row.pop('response', None)
        query = f"""
        Using the numbers {numbers}, create an equation that equals {target}.
        You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
        Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags,
        for example <answer> (1 + 2) / 3 * 4 = 4 </answer>.
        """
        row.update({'target': target, 'query': query})
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        dataset_path='/nas_data/userdata/randy/datasets/countdown-tasks-3to4.jsonl',
        preprocess_func=CoundownTaskPreprocessor()
    )
)
