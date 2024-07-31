import os

from openai import OpenAI
from typing import Optional, Dict, List, Iterator, Union, Literal

from qwen_agent.llm.base import register_llm, BaseChatModel
from qwen_agent.llm.qwen_dashscope import QwenChatAtDS
from qwen_agent.llm.schema import Message
from qwen_agent.llm import QwenChatAtDS

@register_llm('deepseek')
class DeepSeekLLM(BaseChatModel):
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=cfg.get('base_url', "https://api.deepseek.com")
        )

    def _chat_stream(
            self,
            messages: List[Message],
            delta_stream: bool,
            generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        res = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=generate_cfg.get('max_tokens', 1024),
            temperature=generate_cfg.get('temperature', 0.7),
            stream=True
        )
        return res

    def _chat_with_functions(
        self,
        messages: List[Union[Message, Dict]],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        # TODO
        pass

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        # TODO
        pass