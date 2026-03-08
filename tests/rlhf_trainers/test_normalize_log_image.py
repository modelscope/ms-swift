import logging

from swift.rlhf_trainers import utils as rlhf_utils


def test_normalize_log_image_none():
    assert rlhf_utils.normalize_log_image(None) is None


def test_normalize_log_image_empty_list():
    assert rlhf_utils.normalize_log_image([]) is None


def test_normalize_log_image_single_list():
    img = {'path': 'a.png'}
    assert rlhf_utils.normalize_log_image([img]) == img


def test_normalize_log_image_multi_list_warns():
    logger = rlhf_utils.get_logger()
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = ListHandler()
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        img1 = {'path': 'a.png'}
        img2 = {'path': 'b.png'}
        assert rlhf_utils.normalize_log_image([img1, img2]) == img1
    finally:
        logger.removeHandler(handler)

    assert any('Multiple images detected' in record.getMessage() for record in records)


def test_normalize_log_image_dict():
    img = {'path': 'a.png'}
    assert rlhf_utils.normalize_log_image(img) == img


def test_normalize_log_image_string():
    img = 'a.png'
    assert rlhf_utils.normalize_log_image(img) == img
