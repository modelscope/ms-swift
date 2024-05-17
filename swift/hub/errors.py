# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

import requests
from requests.exceptions import HTTPError

from swift.utils.logger import get_logger

logger = get_logger()


class NotSupportError(Exception):
    pass


class NoValidRevisionError(Exception):
    pass


class NotExistError(Exception):
    pass


class RequestError(Exception):
    pass


class GitError(Exception):
    pass


class InvalidParameter(Exception):
    pass


class NotLoginException(Exception):
    pass


class FileIntegrityError(Exception):
    pass


class FileDownloadError(Exception):
    pass


def is_ok(rsp):
    """ Check the request is ok

    Args:
        rsp (Response): The request response body

    Returns:
       bool: `True` if success otherwise `False`.
    """
    return rsp['Code'] == HTTPStatus.OK and rsp['Success']


def _decode_response_error(response: requests.Response):
    if 'application/json' in response.headers.get('content-type', ''):
        message = response.json()
    else:
        message = response.content.decode('utf-8')
    return message


def handle_http_post_error(response, url, request_body):
    try:
        response.raise_for_status()
    except HTTPError as error:
        message = _decode_response_error(response)
        raise HTTPError('Request %s with body: %s exception, '
                        'Response details: %s' % (url, request_body, message)) from error


def handle_http_response(response, logger, cookies, model_id):
    try:
        response.raise_for_status()
    except HTTPError as error:
        if cookies is None:  # code in [403] and
            pass
        message = _decode_response_error(response)
        raise HTTPError('Response details: %s' % message) from error


def raise_on_error(rsp):
    """If response error, raise exception

    Args:
        rsp (_type_): The server response

    Raises:
        RequestError: the response error message.

    Returns:
        bool: True if request is OK, otherwise raise `RequestError` exception.
    """
    if rsp['Code'] == HTTPStatus.OK:
        return True
    else:
        raise RequestError(rsp['Message'])


def datahub_raise_on_error(url, rsp):
    """If response error, raise exception

    Args:
        url (str): The request url
        rsp (HTTPResponse): The server response.

    Raises:
        RequestError: the http request error.

    Returns:
        bool: `True` if request is OK, otherwise raise `RequestError` exception.
    """
    if rsp.get('Code') == HTTPStatus.OK:
        return True
    else:
        raise RequestError(
            f"Url = {url}, Message = {rsp.get('Message')}, Please specify correct dataset_name and namespace.")


def raise_for_http_status(rsp):
    """Attempt to decode utf-8 first since some servers
    localize reason strings, for invalid utf-8, fall back
    to decoding with iso-8859-1.

    Args:
        rsp: The http response.

    Raises:
        HTTPError: The http error info.
    """
    http_error_msg = ''
    if isinstance(rsp.reason, bytes):
        try:
            reason = rsp.reason.decode('utf-8')
        except UnicodeDecodeError:
            reason = rsp.reason.decode('iso-8859-1')
    else:
        reason = rsp.reason

    if 400 <= rsp.status_code < 500:
        http_error_msg = u'%s Client Error: %s for url: %s' % (rsp.status_code, reason, rsp.url)

    elif 500 <= rsp.status_code < 600:
        http_error_msg = u'%s Server Error: %s for url: %s' % (rsp.status_code, reason, rsp.url)

    if http_error_msg:
        req = rsp.request
        if req.method == 'POST':
            http_error_msg = u'%s, body: %s' % (http_error_msg, req.body)
        raise HTTPError(http_error_msg, response=rsp)
