from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
import json5
from LawAgent.SearchEngine import LawDatabase


@register_tool("case_search")
class CaseSearch(BaseTool):
    description = '一个类案查询工具，可以查询某一条类案'
    name = 'case_search'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description':
                '字符串描述需要查询的类案',
            'required': True
        }
    ]
    top_K = 5
    db = LawDatabase()

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            params = json5.loads(params)
        query = str(params['query'])
        labels = params.get('labels', [])
        if not isinstance(labels, list):
            labels = [str(labels)]
        datas = self.search(query, labels, True)

        return datas

    def search(self, query: str, labels: list = None, format_return=False):
        if not labels:
            labels = []
        datas = self.db.search_cases(query, labels, self.top_K)
        if format_return:
            datas = [_["summary"] for _ in datas]
            return f"\n{'#' * 50}\n".join(datas)
        return datas


if __name__ == '__main__':
    print(CaseSearch().search("限定价格", ["横向垄断"], format_return=True))
