# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os

import json
import requests

from swift.version import __version__


# 打标
class ModelTag(object):
    _URL = os.environ.get('MODEL_TAG_URL', None)

    # 模型测试结果
    BATCH_COMMIT_RESULT_URL = f'{_URL}/batchCommitResult'
    # 测试阶段完成
    BATCH_REFRESH_STAGE_URL = f'{_URL}/batchRefreshStage'
    # query_model_stage
    QUERY_MODEL_STAGE_URL = f'{_URL}/queryModelStage'

    HEADER = {'Content-Type': 'application/json'}

    # 检测结果
    MODEL_SKIP = 0
    MODEL_FAIL = 1
    MODEL_PASS = 2

    class ItemResult(object):

        def __init__(self):
            self.result = 0
            self.name = ''
            self.info = ''

        def to_json(self):
            return {'name': self.name, 'result': self.result, 'info': self.info}

    def __init__(self):
        self.job_name = ''
        self.job_id = ''
        self.model = ''
        self.sdk_version = ''
        self.image_version = ''
        self.domain = ''
        self.task = ''
        self.source = ''
        self.stage = ''
        # ItemResult list
        self.item_result = []

    # 发送请求
    def _post_request(self, url, param):
        try:
            logging.info(url + ' query: ' + str(json.dumps(param, ensure_ascii=False)))
            res = requests.post(url=url, headers=self.HEADER, data=json.dumps(param, ensure_ascii=False).encode('utf8'))
            if res.status_code == 200:
                logging.info(f'{url} post结果: ' + res.text)
                res_json = json.loads(res.text)
                if int(res_json['errorCode']) == 200:
                    return res_json['content']
                else:
                    logging.error(res.text)
            else:
                logging.error(res.text)
        except Exception as e:
            logging.error(e)

        return None

    # 提交模型测试结果
    def batch_commit_result(self):
        try:
            param = {
                'sdkVersion':
                self.sdk_version,
                'imageVersion':
                self.image_version,
                'source':
                self.source,
                'jobName':
                self.job_name,
                'jobId':
                self.job_id,
                'modelList': [{
                    'model': self.model,
                    'domain': self.domain,
                    'task': self.task,
                    'itemResult': self.item_result
                }]
            }
            return self._post_request(self.BATCH_COMMIT_RESULT_URL, param)

        except Exception as e:
            logging.error(e)

        return

    # 测试阶段完成
    def batch_refresh_stage(self):
        try:
            param = {
                'sdkVersion': self.sdk_version,
                'imageVersion': self.image_version,
                'source': self.source,
                'stage': self.stage,
                'modelList': [{
                    'model': self.model,
                    'domain': self.domain,
                    'task': self.task
                }]
            }
            return self._post_request(self.BATCH_REFRESH_STAGE_URL, param)

        except Exception as e:
            logging.error(e)

        return

    # 查询模型某个阶段的最新测试结果（只返回单个结果
    def query_model_stage(self):
        try:
            param = {
                'sdkVersion': self.sdk_version,
                'model': self.model,
                'stage': self.stage,
                'imageVersion': self.image_version
            }
            return self._post_request(self.QUERY_MODEL_STAGE_URL, param)

        except Exception as e:
            logging.error(e)

        return None

    # 提交模型UT测试结果
    """
        model_tag = ModelTag()
        model_tag.model = "XXX"
        model_tag.sdk_version = "0.3.7"
        model_tag.domain = "nlp"
        model_tag.task = "word-segmentation"
        item = model_tag.ItemResult()
        item.result = model_tag.MODEL_PASS
        item.name = "ALL"
        item.info = ""
        model_tag.item_result.append(item.to_json())
    """

    def commit_ut_result(self):
        if self._URL is not None and self._URL != '':
            self.job_name = 'UT'
            self.source = 'dev'
            self.stage = 'integration'

            self.batch_commit_result()
            self.batch_refresh_stage()


def commit_model_ut_result(model_name, ut_result):
    model_tag = ModelTag()
    model_tag.model = model_name.replace('damo/', '')
    model_tag.sdk_version = __version__
    # model_tag.domain = ""
    # model_tag.task = ""
    item = model_tag.ItemResult()
    item.result = ut_result
    item.name = 'ALL'
    item.info = ''
    model_tag.item_result.append(item.to_json())
    model_tag.commit_ut_result()
