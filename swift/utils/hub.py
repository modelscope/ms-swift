import os
from typing import Optional

from requests.exceptions import HTTPError

from swift.hub import HubApi, ModelScopeConfig
from swift.hub.constants import ModelVisibility
from .logger import get_logger

logger = get_logger()


def create_ms_repo(hub_model_id: str,
                   hub_token: Optional[str] = None,
                   hub_private_repo: bool = False) -> str:
    assert hub_model_id is not None, 'Please enter a valid hub_model_id'

    api = HubApi()
    if hub_token is None:
        hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
    if hub_token is not None:
        api.login(hub_token)
    visibility = ModelVisibility.PRIVATE if hub_private_repo else ModelVisibility.PUBLIC

    if '/' not in hub_model_id:
        user_name = ModelScopeConfig.get_user_info()[0]
        assert isinstance(user_name, str)
        hub_model_id = f'{user_name}/{hub_model_id}'
        logger.info(
            f"'/' not in hub_model_id, setting hub_model_id: {hub_model_id}")
    try:
        api.create_model(hub_model_id, visibility)
    except HTTPError:
        # The remote repository has been created
        pass
    return hub_model_id
