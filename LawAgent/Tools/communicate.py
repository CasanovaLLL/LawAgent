import os
import re
from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
import json5


@register_tool("communicate")
class Communicate(BaseTool):
    description = '与用户交流，只有通过这个工具产生的消息用户可以看到'
    name = 'communicate'
    parameters = [
        {
            'name': 'message',
            'type': 'string',
            'description':
                '想要发送给用户的信息',
            'required': True
        }
    ]

    def __init__(self, call_user_func):
        super().__init__()
        self.call_user_func = call_user_func

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            try:
                params = json5.loads(params)
            except ValueError:
                pattern = r'[\'"]?message[\'"]?\s*:\s*["\'].*[\'"]\s*}\s*$'
                regex = re.compile(pattern, re.DOTALL)
                text = re.search(regex, params)
                params = {
                    'message': text.group(1)
                }
        query = params['message']
        self.call_user_func(query)
        return '已发送给用户，请等待用户回复'
