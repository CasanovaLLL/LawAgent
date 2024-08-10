from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
import json5
from LawAgent.SearchEngine import LawDatabase


@register_tool("relevant_market_search")
class RelevantMarketSearch(BaseTool):
    description = '一个相关市场检索器，用来检索相关市场的判例'
    name = 'relevant_market_search'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description':
                '描述需要查询的相关市场',
            'required': True
        },
    ]
    top_K = 10
    db = LawDatabase()

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            params = json5.loads(params)
        query = params['query']
        datas = self.search(query, True)

        return datas

    def search(self, query: str, format_return=False):

        datas = self.db.search_relevant_market(query, self.top_K)
        if format_return:
            format_list = []
            for item in datas:
                format_list.append(
                    "\n".join([f"{k}:\n{v.strip()}\n" for k, v in item.items() if 'embedding' not in k and v]))
            return f"\n{'#' * 50}\n".join(format_list)
        return datas


if __name__ == '__main__':
    print(RelevantMarketSearch().search("全球食品巨头", True))
