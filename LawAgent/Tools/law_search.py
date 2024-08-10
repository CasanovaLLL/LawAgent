from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
import json5
from LawAgent.SearchEngine import LawDatabase


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
                '一组法律名称或条数，用来精确匹配，名称与编号应该分开，编号应该采用小写中文。'
                '若设置了这一项一次只能查出一条对应法律'
                '\n例如 ["中华人民共和国反垄断法", "第二十条"]',
            'required': False
        }
    ]
    top_K = 20
    db = LawDatabase()

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            params = json5.loads(params)
        query = params['query']
        labels = params.get('labels', [])
        if not isinstance(labels, list):
            labels = [str(labels)]
        datas = self.search(query, labels, True)

        return datas

    def search(self, query: str, labels: list = None, format_return=False):
        if not labels:
            labels = []
        datas = self.db.search_laws(query, labels, self.top_K)
        if format_return:
            pattern = "{depth}:{code}"
            datas = [pattern.format(**_) for _ in datas]
            return "\n##############\n".join(datas)
        return datas


if __name__ == '__main__':
    print(LawSearch().search("反垄断", ["横向垄断"], True))
