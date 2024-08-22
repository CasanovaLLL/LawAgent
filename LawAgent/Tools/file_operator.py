import os

from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
import json5
from LawAgent.SearchEngine import LawDatabase
from LawAgent.Utils import generate_timestamp, url_add_params
from collections import defaultdict
import json
from threading import Semaphore
import re

work_dir = os.path.join("output/Agent/", generate_timestamp())


@register_tool("notepad")
class NotePad(BaseTool):
    description = '一个记事本，存储了一个dict用来记录关键信息'
    name = 'notepad'
    parameters = [
        {
            'name': 'operation_type',
            'type': 'string',
            'description':
                'Literal["update","read"] 分别对应：更新、查看',
            'required': True
        },
        {
            'name': 'data',
            'type': 'dict',
            'description':
                "当operation_type为 `update`时想写入的信息，为一个dict，会用update方法将这个新的信息写入到共享的信息中。",
            'required': False
        }
    ]

    def __init__(self, note_id=generate_timestamp()):
        super().__init__()
        self.note_id = note_id
        self.data = defaultdict(defaultdict)
        self.file_name = os.path.join(work_dir, f"notepad-{self.note_id}.json")
        self.absolute_path = os.path.abspath(self.file_name)

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            try:
                params = json5.loads(params)
            except ValueError:
                return 'json格式不正确，可以不使用复杂的结构，把数据都加入到一个键里面。'
        query = params['operation_type']
        data = params.get('data', dict())
        try:
            if query == "update":
                self.data.update(data)
                with open(self.file_name, "w") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=4)
                return "Success Update"
            elif query == "read":
                return json.dumps(self.data, ensure_ascii=False, indent=4)
        except Exception as e:
            return f"插入失败，请检查输入。错误为 {e}"

        return r'operation_type 必须是"update","read" 的一种'


@register_tool("file_operator")
class FileOperator(BaseTool):
    description = '一个文件操作工具，可以写入文件以供用户下载。注意，准备好内容后一次性写入。'
    name = 'file_operator'
    parameters = [
        {
            'name': 'operation_type',
            'type': 'string',
            'description':
                'Literal["write","read"] 分别对应：续写、查看',
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

    os.makedirs(work_dir, exist_ok=True)

    def __init__(self, semaphore=Semaphore(),
                 file_path=os.path.join(work_dir, f"agent_out_{generate_timestamp()}.txt"),
                 ):
        super().__init__()
        self.file_path = file_path
        self.semaphore = semaphore

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            try:
                params = json5.loads(params)
                query = params['operation_type']
                text = params.get('text', "")
            except ValueError:
                pattern = r'operation_type.*(write|read|download)'
                func_type = re.search(pattern, params)
                if func_type:
                    func_type = func_type.group(1)
                else:
                    return "错误的参数格式"
                query = func_type
                if func_type == "write":
                    pattern = r'[\'"]?text[\'"]?\s*:\s*["\'].*[\'"]\s*}\s*$'
                    regex = re.compile(pattern, re.DOTALL)
                    text = re.search(regex, params)
                    if text:
                        text = text.group(0)
                    else:
                        return "write模式必须有text参数"
        else:
            query = params['operation_type']
            text = params.get('text', "")
        base_url = os.getenv('DOC_SERVER_URL', 'http://127.0.0.1/file')
        download_link = f'下载链接为 [文书]({url_add_params(base_url, name=self.file_path)})'
        if query == "write":
            with open(self.file_path, "a+") as f:
                f.write(text)
            self.semaphore.release()
            return f"Success Write {download_link}"
        if query == "read":
            with open(self.file_path, "r") as f:
                return f.read() + '\n\n' + download_link
        if query == 'download':
            return download_link
        return r'operation_type 必须是"write","read" 的一种'
