import json
import os

import json5
import requests
import editdistance
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool
from openai import OpenAI, OpenAIError

from LawAgent.Utils import build_username
from tqdm import tqdm


@register_tool('cause_classifier')
class RecursiveClassifier(BaseTool):
    name = 'cause_classifier'
    description = '一个案由分类器，把一个事件分到某一个案由'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description':
            '需要分类的文本',
        'required': True
    }]

    def __init__(self, tree_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deepseek_key = os.environ.get('DEEPSEEK_API_KEY', kwargs.get('deepseek_api_key', None))
        self.base_url = os.environ.get('DEEPSEEK_BASE_URL', kwargs.get('deepseek_base_url', "https://api.deepseek.com"))
        with open(tree_path, 'r', encoding='utf-8') as f:
            self.tree = json5.load(f)

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            params = json5.loads(params)
        prompt = params['query']
        return self.classify(prompt, self.tree["root"])

    def classify(self, prompt: str, node: dict) -> str:
        def _find_nearest_node(node_list: list, query):
            min_distance = 1
            _result = None
            for _ in node_list:
                distance = editdistance.eval(query, _) / max(len(query), len(_))
                if distance < min_distance:
                    min_distance = distance
                    _result = _
            return _result

        # print("Classifier prompt", prompt, node)
        description = node['description']
        next_nodes = node['next']
        if len(next_nodes) == 1:
            return next_nodes[0]
        for _ in range(3):
            try:
                client = OpenAI(
                    api_key=self.deepseek_key,
                    base_url=self.base_url,
                )

                is_multy = node.get('is_multy', False)
                base_prompt = BASE_PROMPT if not is_multy else MULTI_PROMPT
                sys_prompt = base_prompt.format(description=description, choice=next_nodes)
                sys_prompt += BASE_OUTPUT_EXAMPLE if not is_multy else MULTY_OUTPUT_EXAMPLE
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=512,
                    stream=False,
                    response_format={
                        'type': 'json_object'
                    }
                )
                next_node = json5.loads(response.choices[0].message.content)['choice']
                print("Classifier next node", next_node)
                if is_multy:
                    assert isinstance(next_node, list)
                    next_node = [_find_nearest_node(next_nodes, _) for _ in next_node]
                    return next_node
                assert isinstance(next_node, str)
                next_node = _find_nearest_node(next_nodes, next_node)
                if next_node in node['next']:
                    if next_data := self.tree.get(next_node, None):
                        return self.classify(prompt, next_data)
                    return next_node

            except (OpenAIError, AssertionError):
                pass
        return next_nodes[0]


BASE_PROMPT = r"""
你是一个案件分类器，你需要根据提供的事件文本和背景知识，把事件分类到待选案由中的一个。
请输出一个json格式的字符串
背景知识：
{description}
待选案由：
{choice}

"""

MULTI_PROMPT = r"""
你是一个案件分类器，你需要根据提供的事件文本和背景知识，把事件分类到待选案由中的一个或多个。
请输出一个json格式的字符串
背景知识：
{description}
待选案由：
{choice}

"""
BASE_OUTPUT_EXAMPLE = """
输出样例：
{
    "choice":"垄断纠纷"
}
"""
MULTY_OUTPUT_EXAMPLE = """
输出样例：
{
    "choice":["垄断纠纷"]
}
"""


def _description_generate(flatten_tree: dict, reference_dir: str = None):
    from zhipuai import ZhipuAI
    system_prompt = """
    利用搜索引擎，为将一个案件分到以下的分类编写背景知识，主要包括类别的简介和类别间的差异
    - 用精炼简洁的语言
    """
    references = os.listdir(reference_dir) if reference_dir else None

    for k, v in tqdm(flatten_tree.items(), total=len(flatten_tree)):
        try:
            if description := v['description']:
                if references and description in references:
                    with open(os.path.join(reference_dir, description), 'r', encoding='utf-8') as f:
                        description = f.read()
                        v['description'] = description
                continue
            if not v["next"] or len(v['next']) <= 1:
                continue
            query = f"为以下{k}的子案由编写背景知识，请用精炼简洁的语言，同时注意区分不同类别之间的差异：\n{v['next']}"
            print(query)
            client = ZhipuAI(api_key=os.environ["ZHIPU_APIKEY"])
            response = client.chat.completions.create(
                model="glm-4-0520",  # 填写需要调用的模型名称
                messages=[
                    {"role": "system",
                     "content": system_prompt},
                    {"role": "user",
                     "content": query},
                ],
                tools=[{"type": "web_search", "web_search": {"search_result": True}}],
                stream=True,
                top_p=0.7,
                temperature=0.95,
                max_tokens=1024,
            )
            answer_parts = []
            print("--------------------------------------------------------------------------------")
            for chunk in response:
                if content := chunk.choices[0].delta.content:
                    print(content, end="")
                    answer_parts.append(content)
            response_str = "".join(answer_parts)
            print("")
            print(k)
            print("--------------------------------------------------------------------------------")
            v['description'] = response_str
        except Exception as e:
            print(e)
            print("error", k)

    return flatten_tree


if __name__ == '__main__':
    with open("分类树-细.json", "r") as f:
        flatten_tree = json5.load(f)
    flatten_tree = _description_generate(flatten_tree, reference_dir="reference")
    with open("flatten_with_desc.json", "w") as f:
        f.write(json.dumps(flatten_tree, ensure_ascii=False, indent=4))
