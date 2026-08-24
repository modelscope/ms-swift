# Copyright (c) ModelScope Contributors. All rights reserved.
"""Parity tests for migrated dataset registrations: dev's preprocessor against legacy's.

Rows are synthetic and the comparison is offline, on purpose. What a migration can get wrong is the
row transform -- a prompt whose wording drifted, a label read from the wrong column, a filter whose
comparison flipped -- and that is decidable without touching a hub. Whether the dataset still exists
under that id is a different question, and downloading it would answer it slowly and flakily.

Rows are compared with ``None`` values dropped on both sides: legacy's ``_patch_arrow_writer``
unconditionally gave every output an all-null ``images`` / ``objects`` / ``rejected_messages``
column, which dev deliberately does not, and that difference is not a behaviour difference.
"""
from typing import Any, Dict, List

import pytest
from datasets import Dataset as HfDataset


def run(preprocessor, rows: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Map a preprocessor over ``rows``, returning the standard rows with null columns dropped."""
    processed = preprocessor(HfDataset.from_list(rows), load_from_cache_file=False, **kwargs)
    return [{key: value for key, value in row.items() if value is not None} for row in processed]


def run_legacy(preprocessor, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The same, through legacy -- which needs one flag before the comparison is fair.

    ``enable_auto_mapping`` is what turns on legacy's *built-in* alias table (``question`` -> ``query``
    and the rest); without it only explicitly declared renames apply, and legacy's own loader always
    passes it. Dev has no equivalent switch: aliases are the format converter's own knowledge and are
    always in effect.
    """
    return run(preprocessor, rows, enable_auto_mapping=True)


def assert_parity(legacy, dev, rows: List[Dict[str, Any]]) -> None:
    got, want = run(dev, rows), run_legacy(legacy, rows)
    assert got == want, f'\ndev   : {got}\nlegacy: {want}'


# ---- generation from a non-dialogue source ---------------------------------------------------


def test_advertise_gen():
    from swift.dataset.preprocessor.extra import TextGenerationPreprocessor
    from swift.dev.dataset.loader.llm import AdvertiseGenPreprocessor
    legacy = TextGenerationPreprocessor(
        prompt='Task: Generating advertisements based on keywords.\nKeywords: {{QUERY}}\nAdvertisements:',
        columns={
            'content': 'query',
            'summary': 'response'
        })
    rows = [{'content': '类型#裙*版型#显瘦', 'summary': '这款裙子显瘦。'}, {'content': '类型#上衣', 'summary': '好看。'}]
    assert_parity(legacy, AdvertiseGenPreprocessor(), rows)


# ---- classification, in both of its views ----------------------------------------------------


def test_cmnli_as_generation():
    from swift.dataset.preprocessor.extra import ClsGenerationPreprocessor
    from swift.dev.dataset.loader.llm import CmnliPreprocessor
    legacy = ClsGenerationPreprocessor(['neutral', 'entailment', 'contradiction'],
                                       task='Natural Language Inference',
                                       is_pair_seq=True)
    rows = [{
        'sentence1': 'A man is eating.',
        'sentence2': 'A person eats.',
        'label': 1
    }, {
        'sentence1': 'A dog runs.',
        'sentence2': 'Nobody moves.',
        'label': 2
    }]
    assert_parity(legacy, CmnliPreprocessor(), rows)


def test_unlabelled_rows_are_dropped():
    """The test split ships no label; without the answer the row cannot be trained on."""
    from swift.dev.dataset.loader.llm import CmnliPreprocessor
    rows = [{'sentence1': 'a', 'sentence2': 'b', 'label': None}, {'sentence1': 'c', 'sentence2': 'd', 'label': 0}]
    assert len(run(CmnliPreprocessor(), rows)) == 1


def test_jd_as_generation():
    from swift.dataset.preprocessor.extra import ClsGenerationPreprocessor
    from swift.dev.dataset.loader.llm import JdSentimentPreprocessor
    legacy = ClsGenerationPreprocessor(['negative', 'positive'], task='Sentiment Classification', is_pair_seq=False)
    rows = [{'sentence': '质量很差', 'label': 0}, {'sentence': '非常满意', 'label': 1}]
    assert_parity(legacy, JdSentimentPreprocessor(), rows)


def test_jd_as_classification():
    from swift.dataset.preprocessor.core import ClsPreprocessor
    from swift.dev.dataset.loader.llm import JdClsPreprocessor
    legacy = ClsPreprocessor(columns={'sentence': 'query'})
    rows = [{'sentence': '质量很差', 'label': 0}, {'sentence': '非常满意', 'label': 1}]
    assert_parity(legacy, JdClsPreprocessor(), rows)


# ---- embedding and reranking: rows with no assistant turn ------------------------------------


def stsb_rows():
    return [{
        'sentence1': 'A man plays guitar.',
        'sentence2': 'A person is playing an instrument.',
        'score': 0.9
    }, {
        'sentence1': 'A cat sleeps.',
        'sentence2': 'A rocket launches.',
        'score': 0.1
    }]


@pytest.mark.parametrize('legacy_name, dev_name, kwargs', [
    ('StsbPreprocessor', 'StsbPreprocessor', {}),
    ('StsbPreprocessor', 'StsbInfoncePreprocessor', {
        'sim_threshold': 0.75
    }),
    ('StsbGeneratePreprocessor', 'StsbGeneratePreprocessor', {}),
    ('StsbRegressionPreprocessor', 'StsbRegressionPreprocessor', {}),
])
def test_stsb_subsets(legacy_name, dev_name, kwargs):
    """All four views of one dataset, including the threshold that makes the InfoNCE view differ."""
    import swift.dataset.dataset.llm as legacy_module
    import swift.dev.dataset.loader.llm as dev_module
    legacy = getattr(legacy_module, legacy_name)(**kwargs)
    assert_parity(legacy, getattr(dev_module, dev_name)(), stsb_rows())


def test_infonce_view_drops_weak_pairs():
    from swift.dev.dataset.loader.llm import StsbInfoncePreprocessor, StsbPreprocessor
    assert len(run(StsbPreprocessor(), stsb_rows())) == 2
    assert len(run(StsbInfoncePreprocessor(), stsb_rows())) == 1


@pytest.mark.parametrize('as_list', [True, False])
def test_mteb_rerank(as_list):
    """A document column may hold a list or a bare string; both become a list of dialogues.

    The two are separate datasets rather than two rows of one: a single Arrow column cannot hold both
    shapes, which is why the dumps differ per dataset and why the isinstance check exists at all.
    """
    from swift.dataset.dataset.llm import MTEBRerankPreprocessor as LegacyRerank
    from swift.dev.dataset.loader.llm import MTEBRerankPreprocessor

    def shape(*documents):
        return list(documents) if as_list else documents[0]

    rows = [{
        'query': 'what is a transformer',
        'positive': shape('doc one', 'another'),
        'negative': shape('irrelevant')
    }, {
        'query': 'how to sort',
        'positive': shape('quicksort'),
        'negative': shape('a recipe', 'a poem')
    }]
    assert_parity(LegacyRerank(), MTEBRerankPreprocessor(), rows)


# ---- self-cognition: the identity comes from the run, not the data ---------------------------


def cognition_rows():
    return [{
        'query': '你是谁？',
        'response': '我是{{NAME}}，由{{AUTHOR}}训练。',
        'tag': 'zh'
    }, {
        'query': 'Who are you, {{NAME}}?',
        'response': 'I am {{NAME}}, trained by {{AUTHOR}}.',
        'tag': 'en'
    }]


@pytest.mark.parametrize('dev_name, legacy_kwargs', [
    ('SelfCognitionPreprocessor', {}),
    ('Qwen3SelfCognitionPreprocessor', {
        'query_suffix': ' /no_think',
        'response_prefix': '<think>\n\n</think>\n\n'
    }),
    ('EmptyThinkSelfCognitionPreprocessor', {
        'response_prefix': '<think>\n\n</think>\n\n'
    }),
])
def test_self_cognition_subsets(dev_name, legacy_kwargs):
    import swift.dev.dataset.loader.llm as dev_module
    from swift.dataset.dataset.llm import SelfCognitionPreprocessor as LegacyCognition
    legacy = LegacyCognition(**legacy_kwargs)
    legacy.set_name_author(name=('小明', 'Xiaoming'), author=('魔搭', 'ModelScope'))
    dev = getattr(dev_module, dev_name)()
    dev.set_name_author(['小明', 'Xiaoming'], ['魔搭', 'ModelScope'])
    assert_parity(legacy, dev, cognition_rows())


@pytest.mark.parametrize('given, expected', [
    (None, None),
    ('Xiaoming', ('Xiaoming', 'Xiaoming')),
    (['Xiaoming'], ('Xiaoming', 'Xiaoming')),
    (['小明', None], ('小明', '小明')),
    (['小明', 'Xiaoming'], ('小明', 'Xiaoming')),
    ([None, 'Xiaoming'], None),
])
def test_name_normalisation_matches_legacy(given, expected):
    """One value means both languages; legacy did this normalisation in its shared load path."""
    from swift.dev.dataset.loader.llm import SelfCognitionPreprocessor
    assert SelfCognitionPreprocessor.as_language_pair(given) == expected


def test_loader_hands_the_model_identity_to_the_preprocessor():
    """The plumbing, not the substitution: `--model_name` has to reach `build_preprocessor`."""
    from swift.dev.dataset.loader import DatasetInfo
    from swift.dev.dataset.loader.llm import SelfCognitionLoader
    loader = SelfCognitionLoader(
        DatasetInfo(dataset='swift/self-cognition', dataset_type='self-cognition', source='ms'),
        model_name='Xiaoming',
        model_author=['魔搭', 'ModelScope'])
    preprocessor = loader.build_preprocessor(SelfCognitionLoader.subsets[0].resolve(SelfCognitionLoader))
    assert preprocessor.name == ('Xiaoming', 'Xiaoming')
    assert preprocessor.author == ('魔搭', 'ModelScope')


# ---- multimodal rows whose media needs no fetching -------------------------------------------


def test_science_qa():
    """Messages compared against legacy; the image column deliberately is not.

    Legacy built its output row from scratch and never re-attached the media, so a dataset tagged
    multi-modal produced text-only rows. Dev keeps the image, which is what the row is about.
    """
    from swift.dataset.dataset.mllm import ScienceQAPreprocessor as LegacyScienceQA
    from swift.dev.dataset.loader.mllm import ScienceQAPreprocessor
    rows = [{
        'question': 'Which is a mineral?',
        'choices': ['granite', 'quartz'],
        'answer': 1,
        'solution': 'Quartz is a single compound.',
        'image': 'a.jpg'
    }]
    got, want = run(ScienceQAPreprocessor(), rows), run_legacy(LegacyScienceQA(), rows)
    assert got[0]['messages'] == want[0]['messages']
    assert got[0]['images'] == [{'bytes': None, 'path': 'a.jpg'}]
    assert 'images' not in want[0], 'legacy dropped it -- recorded here so the improvement is visible'


def test_voc2007_multilabel():
    from swift.dataset.dataset.mllm import Voc2007MultilabelPreprocessor as LegacyVoc
    from swift.dev.dataset.loader.mllm import Voc2007MultilabelPreprocessor
    rows = [{'webp': 'a.webp', 'npy': [0, 1, 0, 1] + [0] * 16}, {'webp': 'b.webp', 'npy': [1] + [0] * 19}]
    assert_parity(LegacyVoc(columns={'webp': 'images'}), Voc2007MultilabelPreprocessor(), rows)


def test_geometry3k_keeps_the_answer_twice():
    from swift.dataset.dataset.mllm import Geometry3KPreprocessor as LegacyGeometry
    from swift.dev.dataset.loader.mllm import Geometry3KPreprocessor
    rows = [{'problem': 'Find x.', 'answer': '12', 'images': ['a.jpg']}]
    got = run(Geometry3KPreprocessor(), rows)
    assert got[0]['solution'] == '12', 'the reward function reads the reference from `solution`'
    assert_parity(LegacyGeometry(), Geometry3KPreprocessor(), rows)


def test_llava_mix_sft_flattens_content_parts():
    from swift.dataset.dataset.mllm import LLaVAMixSFTPreprocessor as LegacyMix
    from swift.dev.dataset.loader.mllm import LLaVAMixSFTPreprocessor
    rows = [{
        'messages': [{
            'role': 'user',
            'content': [{
                'type': 'image',
                'text': None
            }, {
                'type': 'text',
                'text': 'what is this?'
            }]
        }, {
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': 'a cat'
            }]
        }]
    }]
    assert_parity(LegacyMix(), LLaVAMixSFTPreprocessor(), rows)


def test_coco_detection():
    from swift.dataset.dataset.mllm import CocoPreprocessor as LegacyCoco
    from swift.dev.dataset.loader.mllm import CocoDetectionPreprocessor
    rows = [{
        'images': ['a.jpg'],
        'objects': {
            'category': [0, 15],
            'bbox': [[1, 2, 3, 4], [5, 6, 7, 8]]
        }
    }]
    assert_parity(LegacyCoco(), CocoDetectionPreprocessor(), rows)


def test_clevr_keeps_the_answer_for_the_reward_function():
    from swift.dataset.dataset.mllm import ClevrPreprocessor as LegacyClevr
    from swift.dev.dataset.loader.mllm import ClevrPreprocessor
    rows = [{'problem': 'How many cubes?', 'solution': '3', 'images': ['a.png']}]
    assert run(ClevrPreprocessor(), rows)[0]['solution'] == '3', 'this dataset is tagged grpo'
    assert_parity(LegacyClevr(), ClevrPreprocessor(), rows)


def test_latex_ocr():
    """A dataset with no question column: the prompt is a constant, so only its wording can drift."""
    from swift.dataset.dataset.mllm import LatexocrPreprocessor as LegacyLatexocr
    from swift.dev.dataset.loader.mllm import LatexOcrPreprocessor
    assert_parity(LegacyLatexocr(), LatexOcrPreprocessor(), [{'image': 'a.png', 'text': 'x^2'}])


def test_captcha_images_does_not_keep_a_stray_solution_column():
    """Messages compared against legacy; the extra column legacy leaves behind is not.

    Legacy copied any ``solution`` column aside before its ``map`` and restored it afterwards, so that
    a GRPO reward function could still find it. That fired for *every* dataset with such a column,
    including this one -- which is a captioning dataset where ``solution`` is simply the answer's
    column name, so the copy is the same string twice. Dev keeps it only where it is asked for (see
    :class:`ClevrPreprocessor`).
    """
    from swift.dataset.dataset.mllm import CapchaImagesPreprocessor as LegacyCapcha
    from swift.dev.dataset.loader.mllm import CaptchaPreprocessor
    rows = [{'image': 'a.png', 'solution': 'a1b2'}]
    got = run(CaptchaPreprocessor(), rows)
    want = run_legacy(LegacyCapcha(columns={'solution': 'response'}), rows)
    assert got[0]['messages'] == want[0]['messages']
    assert got[0]['images'] == want[0]['images']
    assert 'solution' not in got[0] and want[0]['solution'] == 'a1b2'


def test_ocrvqa_picks_one_of_the_questions():
    """Not compared against legacy: legacy drew from the global numpy generator, this draws from a
    seeded one, so the two pick different questions by design."""
    from swift.dev.dataset.loader.mllm import OcrvqaPreprocessor
    rows = [{'questions': ['who wrote this?', 'what is the title?'], 'answers': ['A. Smith', 'Dune'], 'image': 'a.jpg'}]
    got = run(OcrvqaPreprocessor(), rows)
    assert len(got) == 1
    query, response = got[0]['messages'][0]['content'], got[0]['messages'][1]['content']
    assert (query, response) in list(zip(rows[0]['questions'], rows[0]['answers'])), 'question and answer must pair up'


def test_ocrvqa_drops_rows_with_no_question():
    from swift.dev.dataset.loader.mllm import OcrvqaPreprocessor
    assert run(OcrvqaPreprocessor(), [{'questions': [], 'answers': [], 'image': 'a.jpg'}]) == []


# ---- grounding ------------------------------------------------------------------------------


def test_grounding_prompt_table_matches_legacy():
    """The wording is the dataset's contribution here, so it is compared verbatim."""
    from swift.dataset.preprocessor.extra import GroundingMixin
    from swift.dev.dataset.loader.mllm import GroundingPreprocessor
    for task, languages in GroundingMixin._grounding_prompts.items():
        for language, prompts in languages.items():
            assert GroundingPreprocessor.PROMPTS[task][language] == prompts


@pytest.mark.parametrize('task', ['grounding', 'caption'])
def test_grounding_prompt_is_drawn_from_the_table(task):
    from swift.dev.dataset.loader.mllm import GroundingPreprocessor
    preprocessor = GroundingPreprocessor.for_task(task)()
    table = [tuple(pair) for prompts in preprocessor.PROMPTS[task].values() for pair in prompts]
    for _ in range(20):
        assert tuple(preprocessor.grounding_prompt()) in table


def test_grit_drops_overlapping_annotations():
    """Two spans covering the same word would mean one word referring to two boxes."""
    from swift.dev.dataset.loader.mllm import GritPreprocessor
    preprocessor = GritPreprocessor.for_task('vqa')()
    rows = [{
        'url': 'a.jpg',
        'caption': 'a dog and a cat',
        'ref_exps': [[0.0, 5.0, 0.1, 0.1, 0.2, 0.2, 0.9], [3.0, 9.0, 0.3, 0.3, 0.4, 0.4, 0.9]]
    }, {
        'url': 'b.jpg',
        'caption': 'a dog and a cat',
        'ref_exps': [[0.0, 5.0, 0.1, 0.1, 0.2, 0.2, 0.9], [6.0, 9.0, 0.3, 0.3, 0.4, 0.4, 0.9]]
    }]
    got = run(preprocessor, rows)
    assert len(got) == 1, 'the overlapping row should be dropped and the disjoint one kept'
    assert got[0]['objects']['ref'] == ['a dog', 'and']
    assert got[0]['messages'][1]['content'] == 'a dog and a cat', 'the vqa view asks for the caption itself'


# ---- rows that are steps of one episode ------------------------------------------------------


def test_mind2web_folds_actions_into_episodes(monkeypatch):
    """Two episodes of two actions each must come back as two rows, not four."""
    from swift.dev.dataset.loader.mllm import Mind2WebPreprocessor
    monkeypatch.setattr('swift.dev.dataset.mm_download.MediaDownloader.safe_save',
                        staticmethod(lambda image, file_name, folder, **kwargs: f'/tmp/{folder}/{file_name}'))
    rows = [{
        'target_action_index': index,
        'action_uid': f'uid{step}{index}',
        'cleaned_html': '<html/>',
        'screenshot': None,
        'confirmed_task': f'task {step}',
        'target_action_reprs': 'the form -> TYPE: hello',
    } for step in range(2) for index in ('0', '1')]

    episodes = list(Mind2WebPreprocessor.episodes(rows))
    assert len(episodes) == 2
    for episode in episodes:
        assert len(episode['messages']) == 4, 'two actions, each a user turn and an assistant turn'
        assert len(episode['images']) == 2
        assert episode['tools'] is Mind2WebPreprocessor.TOOLS
    # Only the action that starts an episode is told what the episode is for.
    assert episodes[0]['messages'][0]['content'].endswith('task 0')
    assert not episodes[0]['messages'][2]['content'].endswith('task 0')
    assert episodes[0]['messages'][1]['content'] == 'Action: TYPE\nAction Input: the form,hello'
