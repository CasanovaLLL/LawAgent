import os
from http import HTTPStatus
from pprint import pformat
from openai import OpenAI
from typing import Optional, Dict, List, Iterator, Union, Literal
import copy
import os
from typing import Dict, Iterator, List, Optional

from qwen_agent.llm.base import register_llm, BaseChatModel, ModelServiceError
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER, Message, FUNCTION
from qwen_agent.llm.schema import Message
from qwen_agent.llm.oai import TextChatAtOAI, OpenAIError


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
        self.tool_calls = []

    def _chat_with_functions(
            self,
            messages: List[Message],
            functions: List[Dict],
            stream: bool,
            delta_stream: bool,
            generate_cfg: dict,
            lang: Literal['en', 'zh'],
            tool_choice: Union[Literal["auto", "none", "required"], Dict] = "auto"
    ) -> Union[List[Message], Iterator[List[Message]]]:
        if delta_stream or stream:
            raise NotImplementedError("DeepSeek function call does not support delta stream")

        messages = [msg.model_dump() for msg in messages]

        if messages[-1]["role"] == FUNCTION:
            messages[-1]["role"] = "tool"
            messages[-1]["tool_call_id"] = self.tool_calls[-1]
            self.tool_calls.pop(-1)

        # 准备工具列表
        tools = [{"type": "function", "function": func} for func in functions]

        try:
            # 调用聊天完成API
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, tools=tools,
                                                  **generate_cfg)

            if response.choices[0].message.tool_calls:
                return_messages = []
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.type == "function":
                        self.tool_calls.append(tool_call.id)
                        return_messages.append(Message(FUNCTION, tool_call.function))
                return return_messages
            return [Message(ASSISTANT, response.choices[0].message.content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)
