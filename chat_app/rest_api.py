import json
import requests
import functools
import logging
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning
from time import time, sleep
from typing import Optional, Dict, Sequence, Any, Union
from random import uniform
import base64
from pydantic import BaseModel, Extra, Field
from typing import Optional

MAX_RETRIES = 10


_BASE_URL = 'https://localhost:8080/'
_USER_NAME = 'admin'
_PASSWORD = 'admin'


class Rest:
    def __init__(self, base_url: str, username: str, password: str,
                 timeout: int = 20, verify: bool = False):
        self.base_url = base_url
        self.timeout = timeout
        self.verify = verify
        self.session = None
        self.server_facts = None
        self.is_tenant_scope = False

        if not verify:
            disable_warnings(InsecureRequestWarning)

        self.login_response: LoginResponse = self.login(username, password)

    def login(self, username: str, password: str) -> bool:
        data = {}
        basic_auth = username + ':' + password
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Basic {}'.format(base64.b64encode(basic_auth.encode()).decode('utf-8'))}
        print(headers)
        session = requests.Session()
        response = session.post(f'{self.base_url}/api/v1.0/login',
                                data=data, timeout=self.timeout, verify=self.verify, headers=headers)
        response.raise_for_status()
        print(response)
        print(response.content)
        json_response = json.loads(response.content)
        print(json_response)
        if b'<html>' in response.content:
            return LoginFailedException(f'Login to {self.base_url} failed, check credentials')
        login_response = LoginResponse(**json_response)
        print(login_response.access_token)
        # _ACCESS_TOKEN = login_response.access_token

        self.session = session
        return login_response

    def _url(self, *path_entries: str) -> str:
        path = '/'.join(path.strip('/') for path in path_entries)

        return f'{self.base_url}/bpa/{path}'


def raise_for_status(response):
    if response.status_code != requests.codes.ok:
        if response.status_code == 429:
            raise ServerRateLimitException('Received rate-limit signal (status-code 429)')

        try:
            reply_data = response.json() if response.text else {}
        except json.decoder.JSONDecodeError:
            reply_data = {'error': {'message': 'Check user permissions'}} if response.status_code == 403 else {}

        details = reply_data.get("error", {}).get("details", "")
        raise RestAPIException(f'{response.reason} ({response.status_code}): '
                               f'{reply_data.get("error", {}).get("message", "Unspecified error message")}'
                               f'{": " if details else ""}{details} [{response.request.method} {response.url}]')


class RestAPIException(Exception):
    """ Exception for REST API errors """
    pass


_access_token = None


def execute_bpa_api(uri: str, path_params: Dict, query_params: Dict):
    try:
        global _access_token
        if _access_token is None:
            with Rest(_BASE_URL, _USER_NAME, _PASSWORD, timeout=30) as api:
                print(f'Login is success {api.login_response.access_token}')
                _access_token = api.login_response.access_token
        headers = {'Authorization': 'Bearer {}'.format(_access_token)}
        print(f'path_params: {path_params}')
        print(f'query_params: {query_params}')
        url = _BASE_URL + (uri.format(**path_params) if path_params else uri)
        print(f'uri {url}')
        print(f'headers {headers}')
        response = requests.get(url=url, params=query_params if query_params else None, headers=headers, verify=False,
                                timeout=30)
        response.raise_for_status()
        print(response)
        print(response.content)
        json_response = json.loads(response.content)
        print(json_response)
        return json_response
    except Exception as e:
        print('error while making api call' +str(e))
        if '401' in str(e):
            ## this exception block code is only for POC.
            with Rest(_BASE_URL, _USER_NAME, _PASSWORD, timeout=30) as api:
                print(f'Login is success {api.login_response.access_token}')
                _access_token = api.login_response.access_token
            return execute_bpa_api(uri,path_params,query_params)
        else:
            return {"statusCode": "E3020","statusMessage": str(e)}

def execute_post_bpa_api(uri: str, data: Dict, path_params: Optional[Dict] = None):
    try:
        global _access_token
        if _access_token is None:
            with Rest(_BASE_URL, _USER_NAME, _PASSWORD, timeout=30) as api:
                print(f'Login is success {api.login_response.access_token}')
                _access_token = api.login_response.access_token
        headers = {'Authorization': 'Bearer {}'.format(_access_token),'Content-Type': 'application/json'}
        print(f'data: {data}')
        url = _BASE_URL + (uri.format(**path_params) if path_params else uri)
        print(f'uri {url}')
        print(f'headers {headers}')
        response = requests.post(url=url, data=data, headers=headers, verify=False,
                                timeout=30)
        response.raise_for_status()
        print(response)
        print(response.content)
        json_response = json.loads(response.content)
        print(json_response)
        return json_response
    except Exception as e:
        print('error while making api call' +str(e))
        if '401' in str(e):
            ## this exception block code is only for POC.
            with Rest(_BASE_URL, _USER_NAME, _PASSWORD, timeout=30) as api:
                print(f'Login is success {api.login_response.access_token}')
                _access_token = api.login_response.access_token
            return execute_post_bpa_api(uri,data,path_params)
        else:
            return {"statusCode": "E3020","statusMessage": str(e)}

if __name__ == '__main__':
    import re

    regexp = '^[\w-]+$'
    print(re.match(regexp, "239A-abcs2a-_Ab"))
    print('abc ')
    # with Rest('https://localhost:8080', 'admin', 'admin', timeout=30) as api:
    #    print(f'Login is success {api.login_response.access_token}')
    dic_query = {'source': 'abc'}
    dic_path = {'source': 'abc_path','path':'test'}
    execute_bpa_api('/api/v1.0/track/source/{source}/{path}', dic_path, dic_query)
