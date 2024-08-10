import os

from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
import json5
from LawAgent.SearchEngine import LawDatabase
from LawAgent.Utils import generate_timestamp


@register_tool("file_operator")
class FileOperator(BaseTool):
    description = '一个文本编辑器,你可以 清空、续写、查看当前大家共同编辑的文本'
    name = 'file_operator'
    parameters = [
        {
            'name': 'operation_type',
            'type': 'string',
            'description':
                'Literal["clear","write","read"] 分别对应，清空、续写、查看三种行为之一',
            'required': True
        },
        {
            'name': 'text',
            'type': 'string',
            'description':
                "当operation_type为 `write`时想写入的文本",
            'required': False
        }
    ]
    work_dir = os.path.join("output/Agent/", generate_timestamp())
    os.makedirs(work_dir, exist_ok=True)
    file_path = os.path.join(work_dir, f"agent_out_{generate_timestamp()}.txt")

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            params = json5.loads(params)
        query = params['operation_type']
        text = params.get('text', "")
        if query == 'clear':
            self.file_path = os.path.join(self.work_dir, f"agent_out_{generate_timestamp()}.txt")
            return "Success Clear"
        if query == "write":
            with open(self.file_path, "a+") as f:
                f.write(text)
            return "Success Write"
        if query == "read":
            with open(self.file_path, "r") as f:
                return f.read()
        return r'operation_type 必须是"clear","write","read" 的一种'
