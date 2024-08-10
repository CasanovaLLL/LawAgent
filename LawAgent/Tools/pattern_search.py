import os

from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
import json5
from LawAgent.SearchEngine import LawDatabase
from LawAgent.Utils import find_nearest_text

__all__ = ['PatternSearch']

PATTERN_DIR = os.getenv("PATTERN_DIR", "data/pattern_dir")
available_pattern = [_ for _ in os.listdir(PATTERN_DIR) if _.endswith(".txt")]
available_pattern_name = [_[:-4] for _ in available_pattern]
PATTERNS = {}

for file_name in available_pattern:
    with open(os.path.join(PATTERN_DIR, file_name), "r") as f:
        PATTERNS[file_name[:-4]] = f.read()


@register_tool("pattern_search")
class PatternSearch(BaseTool):
    description = f'文书模板检索工具，可以检索到 {available_pattern_name} 中的一个'
    name = 'pattern_search'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description':
                '需要查询的模板名称，一次只能查一个',
            'required': True
        }
    ]

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            params = json5.loads(params)
        query = params['query']
        datas = self.search(query)

        return datas

    def search(self, query):
        target = find_nearest_text(available_pattern_name, query)
        try:
            return PATTERNS[target]
        except:
            KeyError(f"应该从{available_pattern_name} 选一个查看")

if __name__ == '__main__':
    print(PatternSearch().search("公司诉讼公司"))
