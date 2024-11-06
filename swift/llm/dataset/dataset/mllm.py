from typing import Any, Dict

from ..constant import MLLMDatasetName
from ..media import MediaResource
from ..preprocess import MessagesPreprocessor, ResponsePreprocessor, RowPreprocessor
from ..register import DatasetMeta, SubsetDataset, register_dataset


class GPT4vDataset(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = 'What is the caption of this image?'
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        MLLMDatasetName.gpt4v_dataset,
        ms_dataset_id='swift/gpt4v-dataset',
        hf_dataset_id='laion/gpt4v-dataset',
        preprocess_func=GPT4vDataset(columns_mapping={
            'link': 'images',
            'caption': 'response'
        }),
        subsets=['default'],
        split=['train'],
        tags=['en', 'caption', 'multi-modal', 'quality'],
    ))

register_dataset(
    DatasetMeta(
        MLLMDatasetName.rlaif_v,
        ms_dataset_id='swift/RLAIF-V-Dataset',
        hf_dataset_id='openbmb/RLAIF-V-Dataset',
        subsets=['default'],
        preprocess_func=ResponsePreprocessor(columns_mapping={
            'image': 'images',
            'question': 'query',
            'chosen': 'response',
            'rejected': 'rejected_response'
        }),
        tags=['rlhf', 'dpo', 'multi-modal', 'en'],
    ))


class LLaVADataPreprocessor(MessagesPreprocessor):

    modals = ['image']

    def prepare_dataset(self, dataset):
        self.all_folders = {}
        for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
            self.all_folders[media_type] = MediaResource.download(media_type)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not row['images']:
            return {}
        row = super().preprocess(row)
        images = [p['path'] for p in row['images']]
        new_images = []
        for image in images:
            if 'coco/' in image:
                image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
            elif 'gqa/' in image:
                image = os.path.join(self.all_folders['gqa'], image.replace('gqa/', ''))
            elif 'ocr_vqa/' in image:
                image = os.path.join(self.all_folders['ocr_vqa'], image)
            elif 'textvqa/' in image:
                image = os.path.join(self.all_folders['textvqa'], image.replace('textvqa/', ''))
            elif 'VG_100K/' in image:
                image = os.path.join(self.all_folders['VG_100K'], image.replace('vg/', ''))
            elif 'VG_100K_2/' in image:
                image = os.path.join(self.all_folders['VG_100K_2'], image.replace('vg/', ''))
            new_images.append(image)
        if all(os.path.exists(image) for image in new_images):
            row['images'] = new_images
        else:
            row['images'] = []
        return row


register_dataset(
    DatasetMeta(
        MLLMDatasetName.llava_data_instruct,
        ms_dataset_id='swift/llava-data',
        hf_dataset_id='TIGER-Lab/llava-data',
        subsets=['llava_instruct'],
        preprocess_func=LLaVADataPreprocessor(),
        split=['train'],
        tags=['sft', 'multi-modal', 'quality'],
    ))


def preprocess_mind2web(dataset, **kwargs):

    def preprocess_row(row: Dict[str, Any]) -> Dict[str, Any]:
        raw_html = row['cleaned_html']
        screenshot = row['screenshot']
        row['screenshot'] = MediaResource.safe_save(screenshot, row['action_uid'] + '.jpg', 'mind2web')
        action = row['target_action_reprs']
        actions = action.split('->')
        row['query'] = f'The snapshot of screen:<image>\nThe html source code:{raw_html}\n'
        action = actions[-1]
        where = actions[0] if len(actions) > 1 else ''
        what = ''
        if ':' in action:
            action, what = action[:action.find(':')], action[action.find(':') + 1:]
        row['response'] = f'Action: {action.strip()}\nAction Input: {where.strip()}{"," + what.strip()}'
        return row

    conversations = []
    tools = [{
        'function': {
            'name': 'CLICK',
            'desc': 'Choose and click an element in the web page',
            'parameter': [{
                'element': 'string, the element in the web page to click'
            }]
        }
    }, {
        'function': {
            'name':
            'TYPE',
            'desc':
            'Input some text into a web element like <input> or <textbox>',
            'parameter': [{
                'element': 'string, the element in the web page to input to',
                'content': 'string, what content to input into the textbox elment'
            }]
        }
    }, {
        'function': {
            'name':
            'SELECT',
            'desc':
            'Select an element from a combobox',
            'parameter': [{
                'element': 'string, the combobox or dropdown in the web page on which the select happens',
                'content': 'string, which choices to choose'
            }]
        }
    }]

    def history_to_messages(history):
        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})
        return messages

    if isinstance(dataset, HfIterableDataset):

        def generate_example(dataset):
            history = []
            images = []
            for row in dataset:
                target_action_index = row['target_action_index']
                row = preprocess_row(row)
                query = row['query']
                if target_action_index == '0':
                    if history:
                        yield {'messages': history_to_messages(history), 'images': images, 'tools': tools}
                        images = []
                        history = []
                    query = query + '\n' + row['confirmed_task']
                history.append([query, row['response']])
                images.append(row['screenshot'])

            if history:
                yield {'messages': history_to_messages(history), 'images': images, 'tools': tools}

        return HfIterableDataset.from_generator(generate_example, gen_kwargs={'dataset': dataset})

    history = []
    images = []
    for row in tqdm(dataset):
        target_action_index = row['target_action_index']
        row = preprocess_row(row)
        query = row['query']
        if target_action_index == '0':
            if history:
                conversations.append({'messages': history_to_messages(history), 'images': images, 'tools': tools})
                images = []
                history = []
            query = query + '\n' + row['confirmed_task']
        history.append([query, row['response']])
        images.append(row['screenshot'])

    if history:
        conversations.append({'messages': history_to_messages(history), 'images': images, 'tools': tools})

    return HfDataset.from_list(conversations)


register_dataset(
    DatasetMeta(
        MLLMDatasetName.mind2web,
        ms_dataset_id='swift/Multimodal-Mind2Web',
        hf_dataset_id='osunlp/Multimodal-Mind2Web',
        preprocess_func=preprocess_mind2web,
        tags=['agent', 'multi-modal']))
