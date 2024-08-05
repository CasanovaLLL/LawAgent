import os
from http import HTTPStatus
from pprint import pformat
from openai import OpenAI
from typing import Optional, Dict, List, Iterator, Union, Literal
import copy
import os
from pprint import pformat
from typing import Dict, Iterator, List, Optional

import openai

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.text_base import BaseTextChatModel
from qwen_agent.log import logger

from qwen_agent.llm.base import register_llm, BaseChatModel, ModelServiceError
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER, Message
from qwen_agent.llm.schema import Message
from qwen_agent.llm.oai import TextChatAtOAI


@register_llm('DeepSeek')
class DeepSeekLLM(TextChatAtOAI):
    def __init__(self, cfg: Optional[Dict] = None):
        deepseek_key = os.getenv("DEEPSEEK_KEY")
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = self.model or 'deepseek-chat'
        deepseek_cfg = {
            "api_key": deepseek_key,
            "base_url": deepseek_base_url,
        }
        cfg.update(deepseek_cfg)
        super().__init__(cfg)

    def _chat_with_functions(
            self,
            messages: List[Message],
            functions: List[Dict],
            stream: bool,
            delta_stream: bool,
            generate_cfg: dict,
            lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        raise NotImplementedError
