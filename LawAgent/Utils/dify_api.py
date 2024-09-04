import os

import requests
import json5


class DifyAPI:
    def __init__(self, api_key: str = None, api_url: str = None):
        self.api_key = api_key or os.getenv("DIFY_API_KEY")
        self.api_url = api_url or os.getenv("DIFY_BASE_URL")
        self.conversation_id = None

    def conversation(self, query: str):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        # 完整的请求体
        body = {
            "inputs": {},
            "query": query,
            "response_mode": "streaming",
            "user": "abc-123",
            "files": []
        }
        if self.conversation_id:
            body["conversation_id"] = self.conversation_id

        # 发送请求并处理流式响应
        response = requests.post(self.api_url, headers=headers, json=body, stream=True)

        if response.status_code != 200:
            response.raise_for_status()

        # 处理流式响应
        collected_data = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith('data:'):
                    chunk = line[len('data:'):].strip()
                    if chunk:
                        chunk_data = chunk
                        try:
                            chunk_dict = json5.loads(chunk_data)  # 将字符串转换为字典
                            if chunk_dict.get('event') == 'message':
                                collected_data += chunk_dict.get('answer', '')
                            elif chunk_dict.get('event') == 'message_end':
                                self.conversation_id = chunk_dict.get('conversation_id')
                        except Exception as e:
                            print(f"Error parsing chunk: {chunk_data}, error: {e}")

        return collected_data
