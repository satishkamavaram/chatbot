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


class LoginResponse(BaseModel):
    token_type: Optional[str]
    access_token: Optional[str]
    refresh_token: Optional[str]
    expires_in: Optional[int]
    expires_in_refresh: Optional[int]
    auth_mode: Optional[str]
    group_auth: Optional[bool]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def backoff_wait_secs(retry_count: int, ceiling: int = 5, variance: float = 0.25) -> float:
    return (1 << min(retry_count, ceiling)) * (1 + uniform(-variance, variance))


def backoff_retry(fn):
    @functools.wraps(fn)
    def retry_fn(*args, **kwargs):
        for retry in range(MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except ServerRateLimitException as ex:
                wait_secs = backoff_wait_secs(retry)
                logging.getLogger(__name__).debug(f'{ex}: Retry {retry + 1}/{MAX_RETRIES}, backoff {wait_secs:.3}s')
                sleep(wait_secs)
        else:
            raise RestAPIException(f'Maximum retries exceeded ({MAX_RETRIES})')

    return retry_fn


_BASE_URL = 'https://bpa-adhoc4.cisco.com/bpa'
#_BASE_URL = 'https://bpa-beetles4.cisco.com/bpa'
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if self.session is not None:
        # In 19.3, logging out actually de-authorize all sessions a user may have from the same IP address. For
        # instance browser windows open from the same laptop. This is fixed in 20.1.
        # self.logout()
        # self.session.close()

        return False

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

    def logout(self) -> bool:
        response = self.session.get(f'{self.base_url}/logout', params={'nocache': str(int(time()))})
        return response.status_code == requests.codes.ok

    @property
    def server_version(self) -> str:
        return self.server_facts.get('platformVersion', '0.0')

    @property
    def is_multi_tenant(self) -> bool:
        return self.server_facts.get('tenancyMode', '') == 'MultiTenant'

    @property
    def is_provider(self) -> bool:
        return self.server_facts.get('userMode', '') == 'provider'

    @backoff_retry
    def get(self, *path_entries: str, **params: Union[str, int]) -> Dict[str, Any]:
        response = self.session.get(self._url(*path_entries),
                                    params=params if params else None,
                                    timeout=self.timeout, verify=self.verify)
        raise_for_status(response)
        return response.json()

    @backoff_retry
    def post(self, input_data: Dict[str, Any], *path_entries: str) -> Union[Dict[str, Any], None]:
        # With large input_data, vManage fails the post request if payload is encoded in compact form. Thus encoding
        # with indent=1.
        response = self.session.post(self._url(*path_entries), data=json.dumps(input_data, indent=1),
                                     timeout=self.timeout, verify=self.verify)
        raise_for_status(response)

        # POST may return an empty string, return None in this case
        return response.json() if response.text else None

    @backoff_retry
    def put(self, input_data: Dict[str, Any], *path_entries: str) -> Union[Dict[str, Any], None]:
        # With large input_data, vManage fails the put request if payload is encoded in compact form. Thus encoding
        # with indent=1.
        response = self.session.put(self._url(*path_entries), data=json.dumps(input_data, indent=1),
                                    timeout=self.timeout, verify=self.verify)
        raise_for_status(response)

        # PUT may return an empty string, return None in this case
        return response.json() if response.text else None

    @backoff_retry
    def delete(self, *path_entries: str, input_data: Optional[Dict[str, Any]] = None,
               **params: str) -> Union[Dict[str, Any], None]:
        response = self.session.delete(self._url(*path_entries),
                                       data=json.dumps(input_data, indent=1) if input_data is not None else None,
                                       params=params if params else None,
                                       timeout=self.timeout, verify=self.verify)
        raise_for_status(response)

        # DELETE normally returns an empty string, return None in this case
        return response.json() if response.text else None

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


def is_version_newer(version_1: str, version_2: str) -> bool:
    def parse(version_string: str) -> Sequence[int]:
        # Development versions may follow this format: '20.1.999-98' or '20.9.0.02-li'
        return [int(v) for v in f"{version_string}.0".split('.')[:2]]

    return parse(version_2) > parse(version_1)


def response_id(response_payload: Dict[str, str]) -> str:
    if response_payload is not None:
        for value in response_payload.values():
            return value

    raise RestAPIException("Unexpected response payload")


class RestAPIException(Exception):
    """ Exception for REST API errors """
    pass


class LoginFailedException(RestAPIException):
    """ Login failure """
    pass


class BadTenantException(RestAPIException):
    """ Provided tenant is invalid or not applicable """
    pass


class ServerRateLimitException(RestAPIException):
    """ REST API server is rate limiting the request via 429 status code """
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
    # with Rest('https://bpa-adhoc4.cisco.com', 'admin', 'admin', timeout=30) as api:
    #    print(f'Login is success {api.login_response.access_token}')
    dic_query = {'source': 'abc'}
    dic_path = {'source': 'abc_path','path':'test'}
    execute_bpa_api('/api/v1.0/track/source/{source}/{path}', dic_path, dic_query)
