import json5

from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('recursive_classifier')
class RecursiveClassifier(BaseTool):
    name = 'recursive_classifier'
    description = '一个递归分类器，将输入按照一个分类树分到某一个子类'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description':
            '需要分类的文本',
        'required': True
    }]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        prompt = json5.loads(params)['prompt']
