from qwen_agent.agents import Assistant

import os

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import Message
from typing import Union
import json5
from LawAgent.Tools.file_operator import FileOperator
from threading import Semaphore
from queue import Queue


@register_tool("article_writer")
class ArticleWriter(BaseTool):
    description = '让文章编写专员编写一篇文章'
    name = 'article_writer'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description':
                '描述需要写的文章',
            'required': True
        }
    ]

    def __init__(self, note_path: str, semaphore=Semaphore(), llm=None, queue=Queue(1)):
        super().__init__()
        self.semaphore = semaphore
        self.note_path = note_path
        self.llm = llm
        self.queue = queue

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        if isinstance(params, str):
            params = json5.loads(params)

        query = params['query']
        with open(self.note_path, 'r', encoding='utf-8') as f:
            data = f.read()
        messages = [
            Message(role='user', content=f'完成一篇关于{query}的文书')
        ]
        file_semaphore = Semaphore(0)
        file = FileOperator(file_semaphore)
        writer = Assistant(name='文书编写者',
                           description='你是一位反垄断方面的文书编写专家。\n在文本编辑器中编写文书。',
                           llm=self.llm,
                           function_list=[file]
                           )
        while True:
            *_, response = writer.run(messages, lang='zh', knowledge=f'已有的数据为{data}')
            messages.extend(response)
            print("文书编写者", response)
            self.queue.put(response)
            if file_semaphore.acquire(blocking=False):
                break
        return file.call({'operation_type': 'download'})
