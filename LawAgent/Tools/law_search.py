from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
import json5


@register_tool("law_search")
class LawSearch(BaseTool):
    description = '一个法律查询工具，可以查询某一条法律'
    name = 'law_search'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description':
                '需要查询的法律内容',
            'required': True
        },
        {
            'name': 'labels',
            'type': 'list[str]',
            'description':
                '一组法律名称或条数，用来精确匹配，名称与编号应该分开，编号应该采用小写中文'
                '例如 ["中华人民共和国反垄断法", "第二十条"]',
            'required': False
        }
    ]

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            params = json5.loads(params)
        query = params['prompt']
        return self.classify(prompt, self.tree["root"])
