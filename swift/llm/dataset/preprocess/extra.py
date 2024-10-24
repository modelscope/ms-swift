from copy import copy
from typing import Any, Dict, List, Optional

import numpy as np

from swift.llm import Messages
from swift.utils import get_logger
from .core import DATASET_TYPE, ResponsePreprocessor, RowPreprocessor

logger = get_logger()


class GroundingMixin:
    """This class offers prompts to the grounding task"""
    task_type: Optional[str] = None

    _grounding_language_mixin = [0.8, 0.2]
    _grounding_prompts = {
        'grounding': {
            'en': [('<ref-object>', '<bbox>'), ('The positions of <ref-object> is', '<bbox>'),
                   ('Find the positions of <ref-object>', '<bbox>'), ('Where is <ref-object>', '<bbox>'),
                   ('Find <ref-object>', '<bbox>'), ('Show me <ref-object>', '<bbox>'),
                   ('Detect <ref-object>', '<bbox>'), ('Locate <ref-object>', '<bbox>'),
                   ('Tell me the location of <ref-object>', '<bbox>'), ('Give the location of <ref-object>', '<bbox>'),
                   ('Provide the bounding box coordinate of <ref-object>', '<bbox>')],
            'zh': [('<ref-object>', '<bbox>'), ('<ref-object>的位置在图片中', '<bbox>'), ('<ref-object>在图片中', '<bbox>'),
                   ('<ref-object>在', '<bbox>'), ('找到<ref-object>的位置', '<bbox>'), ('<ref-object>在哪里', '<bbox>'),
                   ('提供<ref-object>的坐标位置', '<bbox>')]
        },
        'caption': {
            'en': [
                ('<bbox>', '<ref-object>'),
                ('The object at position <bbox>', '<ref-object>'),
                ('This <bbox> is', '<ref-object>'),
                ('What is the object at <bbox>', '<ref-object>'),
                ('Describe <bbox>', '<ref-object>'),
                ('<bbox> is', '<ref-object>'),
                ('The bounding box coordinate <bbox> contains', '<ref-object>'),
            ],
            'zh': [
                ('<bbox>', '<ref-object>'),
                ('<bbox>是什么', '<ref-object>'),
                ('<bbox>的位置包含', '<ref-object>'),
                ('描述<bbox>', '<ref-object>'),
                ('<bbox>中是', '<ref-object>'),
                ('坐标<bbox>描述了什么', '<ref-object>'),
                ('描述<bbox>中的事物', '<ref-object>'),
            ]
        },
    }

    def construct_grounding_prompt(self):
        # TODO Only support one bbox to one object
        lang = np.random.choice(['en', 'zh'], p=[0.8, 0.2])
        prompts = GroundingMixin._grounding_prompts[self.task_type][lang]
        query, response = prompts[np.random.choice(range(len(prompts)))]
        return query, response


multimodal_tags = {
    'image': '<image>',
    'audio': '<audio>',
    'video': '<video>',
}
multimodal_keys = {
    'audio': 'audios',
    'image': 'images',
    'video': 'videos',
}


# TODO
class ExtraRowPreprocessor(RowPreprocessor, GroundingMixin):
    """
    This class owns the data processing scenario.
    """

    has_tool: bool = False
    columns_mapping: Dict[str, str] = {}
    modals: List[str] = []  # image/video/audio
    modal_tags: List[str] = {}
    modal_keys: List[str] = {}

    def __init__(self, **kwargs):
        if 'has_tool' in kwargs:
            self.has_tool = kwargs.pop('has_tool')
        if 'columns_mapping' in kwargs:
            self.columns_mapping = kwargs.pop('columns_mapping')
        if 'modals' in kwargs:
            self.modals = kwargs.pop('modals')
        if 'modal_tags' in kwargs:
            self.modal_tags = kwargs.pop('modal_tags')
        if 'modal_keys' in kwargs:
            self.modal_keys = kwargs.pop('modal_keys')
        if self.modals:
            if not self.modal_tags:
                self.modal_tags = {modal: multimodal_tags[modal] for modal in self.modals}
            if not self.modal_keys:
                self.modal_keys = {modal: multimodal_keys[modal] for modal in self.modals}

    def replace_standard_tag(self, messages: Messages, medias: List[Any], modal: str):
        """Replace tags to standard tags,
            for example:
            query: <img>What is in the image?
            to:
            query: <image>What is in the image?
            Meanwhile, if the conversation shorts of <image>,
            this method will add equal ones to the head of the messages.
        Args:
            messages: The messages input
            medias: The medias, like images or videos
            modal: The modal, like image/video/audio
        Returns:
            Messages
        """
        assert len(self.modal_tags) == len(self.modals)
        _modal_tag = self.modal_tags[modal]
        assert _modal_tag is not None
        media_cnt = len(medias) if isinstance(medias, (tuple, list)) else 1 if medias else 0
        # like <image>, etc
        standard_tag = multimodal_tags[modal]
        all_content = ''.join([m['content'] for m in messages])
        if _modal_tag in all_content:
            # If the messages already have placeholders like `<image>`
            assert all_content.count(_modal_tag) == media_cnt
            for m in messages:
                # Replace to standard tag
                m['content'] = m['content'].replace(_modal_tag, standard_tag)
        elif '<img>' in all_content and '</img>' in all_content:
            pass
        else:
            for m in messages:
                if m['role'] not in ('tool', 'system', 'assistant'):
                    m['content'] = ''.join([standard_tag] * media_cnt) + m['content']
                    break

        return messages

    def query_to_message(self, row: Dict[str, Any]):
        """A compatible method to turn query/response to messages, this is used to fit the existing dataset_info.json"""
        messages = []
        if 'query' in row:
            query = row['query']
            if isinstance(query, list):
                query = query[np.random.choice(range(len(query)))]
            messages.append({'role': 'user', 'content': query})
        if 'response' in row:
            response = row['response']
            if isinstance(response, list):
                response = response[np.random.choice(range(len(response)))]
            messages.append({'role': 'assistant', 'content': response})
        old_messages = row.get('messages') or []
        old_messages.extend(messages)
        row['messages'] = old_messages

    def parse_media_from_row(self, row: Dict[str, Any], modal: str):
        """Parse media from a row
        Args:
            row: The row in `Dict`
            modal: The modal
        """
        modal_key = self.modal_keys[modal]
        if isinstance(modal_key, str):
            if modal_key in row:
                medias = row[modal_key]
            else:
                medias = None
        elif modal_key:  # function
            medias = modal_key(row)
        else:
            medias = None
        return medias

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Please override this method"""
        return row

    def empty_row(self) -> Dict[str, Any]:
        """Generate an empty row, for later filtering"""
        row = {'messages': None}
        if self.has_tool:
            row['tools'] = None
        for _modal in self.modals:
            row[multimodal_keys[_modal]] = None
        return row

    def filter(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Filter unwanted row, by default `messages` must exist"""
        return row.get('messages')

    def rename_columns(self, dataset: DATASET_TYPE, columns_mapping: Dict[str, str]) -> DATASET_TYPE:
        """Rename columns"""
        return dataset.rename_columns(columns_mapping)

    def prepare_multi_modal(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare multi-modal.
        1. Replace tag to standard ones of all modals
        2. Construct grounding prompt and set them
        Args:
            row: The row
        Returns:
            The output dict
        """
        for modal in self.modals:
            medias = self.parse_media_from_row(row, modal)
            if 'messages' not in row:
                row['messages'] = []
            if self.task_type in self._grounding_prompts.keys():
                query, response = self.construct_grounding_prompt()
                row['messages'].extend([
                    {
                        'role': 'user',
                        'content': query
                    },
                    {
                        'role': 'assistant',
                        'content': response
                    },
                ])
            if medias:
                row['messages'] = self.replace_standard_tag(row['messages'], medias, modal)
                modal_key = self.modal_keys[modal]
                if not isinstance(modal_key, str):
                    row[multimodal_keys[modal]] = medias
                else:
                    row[modal_key] = medias
        return row

    def prepare_downloading(self, dataset: DATASET_TYPE) -> None:
        """Override this method to prepare extra resources downloading"""
        pass

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
        """Preprocess a dataset.
        Args:
            dataset: The dataset to be mapped and filtered.
            **kwargs:
                Extra kwargs for mapping&filtering
        Returns:
            The processed dataset
        """
        maybe_multi_modal_keys = ['image', 'images', 'audio', 'audios', 'video', 'videos']
        maybe_multi_modal = any([key in dataset.features for key in maybe_multi_modal_keys])
        self.prepare_downloading(dataset)

        columns_mapping = copy(self.columns_mapping)
        # Replace un-standard media keys to standard keys
        for idx, _modal in enumerate(self.modals):
            modal_key = self.modal_keys[_modal]
            standard_key = multimodal_keys[_modal]
            if standard_key not in dataset.features:
                columns_mapping[modal_key] = standard_key

        if columns_mapping:
            dataset = self.rename_columns(dataset, columns_mapping)

        dataset = dataset.map(self.preprocess, num_proc=num_proc, load_from_cache_file=load_from_cache_file)
        dataset = dataset.filter(self.filter, **kwargs)

        all_keys = list(multimodal_keys.values())
        if (maybe_multi_modal and not self.modals) or (maybe_multi_modal
                                                       and not any([key in dataset.features for key in all_keys])):
            logger.warn('FOUND MULTI MODAL IN DATASETS, BUT NO KEY IN DATASET, MAYBE THE DATASET IS NOT CORRECT')
        if self.modals and not any([key in dataset.features for key in all_keys]):
            raise ValueError('Modals are set and no media keys set')
        return dataset


class TextGenerationPreprocessor(ResponsePreprocessor):

    def __init__(self,
                 *,
                 prompt: str,
                 query_tag: str = '{{QUERY}}',
                 columns_mapping: Optional[Dict[str, str]] = None,
                 remove_useless_columns: bool = True) -> None:
        self.query_tag = query_tag
        self.prompt = prompt
        super().__init__(columns_mapping=columns_mapping, remove_useless_columns=remove_useless_columns)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = super().preprocess(row)
        messages = row['messages']
        query_message = messages[-2]
        query_message['content'] = self.prompt.replace(self.query_tag, query_message['content'])
        return row


class ClsPreprocessor(ResponsePreprocessor):

    def __init__(self,
                 labels: List[str],
                 *,
                 task: str,
                 is_pair_seq: bool = False,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 remove_useless_columns: bool = True) -> None:
        self.labels = labels
        self.task = task
        self.is_pair_seq = is_pair_seq

        category = ', '.join(labels)
        self.sentence2_key = 'sentence2'
        self.label_key = 'label'
        if is_pair_seq:
            self.sentence_key = 'sentence1'
            inputs = 'Sentence1: {sentence1}\nSentence2: {sentence2}'
        else:
            self.sentence_key = 'sentence'
            inputs = 'Sentence: {sentence}'
        self.prompt = f"""Task: {task}
{inputs}
Category: {category}
Output:"""
        super().__init__(columns_mapping=columns_mapping, remove_useless_columns=remove_useless_columns)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        label = row.pop(self.label_key, None)
        if label is None:
            return

        if self.is_pair_seq:
            query = self.prompt.format(sentence1=row.pop(self.sentence_key), sentence2=row.pop(self.sentence2_key))
        else:
            query = self.prompt.format(sentence=row.pop(self.sentence_key))
        row['query'] = query
        row['response'] = self.labels[int(label)]
        return super().preprocess(row)
