import queue
import time

import gradio as gr

import modelscope_studio as mgr

from modelscope_studio.components.Chatbot.llm_thinking_presets import qwen

from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot

from threading import Semaphore
from qwen_agent.llm.schema import CONTENT, FILE, IMAGE, NAME, ROLE, USER, Message, ASSISTANT

from LawAgent.Utils import get_avatar
from queue import Queue
from pydantic import BaseModel
from typing import Any
import time


class ChatFuncData(BaseModel):
    semaphore: Any
    chat_messages: Any
    agent: Any
    agent_list: Any
    writer_queue: Any


state_dict = dict()


class MyWebUI:
    @staticmethod
    def run(build_agent_func):
        with gr.Blocks() as main:
            messages = gr.State(value=[])
            id_state = gr.State()
            with gr.Row():
                with gr.Column():
                    chat_block = mgr.Chatbot(
                        label="对话",
                        value=convert_history_to_chatbot([{
                            ROLE: ASSISTANT,
                            CONTENT: [{
                                'text': "您好,请介绍你想要撰写的文书的信息"
                            }],
                        }]),
                        llm_thinking_presets=[qwen()],
                        height=600,
                        avatar_images=[
                            {
                                'name': "user",
                                'avatar': get_avatar(0)
                            },
                            {
                                'name': "反垄断文书撰写团队",
                                'avatar': get_avatar(4)
                            }

                        ]
                    )
                    textbox = gr.Textbox()
                    button = gr.Button("Send")

                with gr.Column():
                    thought_block = mgr.Chatbot(
                        label="内部讨论",
                        value=convert_history_to_chatbot(messages.value),
                        llm_thinking_presets=[qwen()],
                        height=800,
                        avatar_images=[
                            get_avatar(0),
                            [
                                {
                                    'name': agent_name,
                                    'avatar': get_avatar(i + 1)
                                } for i, agent_name in
                                enumerate(['文书模板确定者', '反垄断法律专家', '案例检索者', '部门主管'])
                            ]]
                    )

                with gr.Column():
                    writer_block = mgr.Chatbot(
                        label="文书编写者",
                        llm_thinking_presets=[qwen()],
                        height=800
                    )

                button.click(fn=MyWebUI.chat(build_agent_func), inputs=[textbox,
                                                                        chat_block,
                                                                        thought_block,
                                                                        messages,
                                                                        id_state],
                             outputs=[textbox,
                                      chat_block,
                                      thought_block,
                                      messages,
                                      button,
                                      id_state])
                writer_block.attach_load_event(MyWebUI.set_writer(), 1, [writer_block, id_state])
        return main

    @staticmethod
    def set_writer():
        global state_dict

        def _set_writer(writer_block, state):
            try:
                state_data = state_dict[state]
                writer_queue = state_data.writer_queue
                responses = writer_queue.get(block=False)
                display_responses = convert_fncall_to_text(responses)
                num_output_bubbles = 0
                # print(display_responses)
                if not display_responses:
                    return
                if display_responses[-1][CONTENT] is None:
                    return
                while len(display_responses) > num_output_bubbles:
                    writer_block.append([None, None])
                    num_output_bubbles += 1
                for i, rsp in enumerate(display_responses):
                    # print(rsp)
                    writer_block[i - len(display_responses)][1] = rsp[CONTENT]

            except queue.Empty:
                pass
            except KeyError:
                pass
            return writer_block

        return _set_writer

    # noinspection PyTypeChecker
    @staticmethod
    def chat(build_agent_func):
        global state_dict

        def _chat(_input,
                  chat_block: mgr.Chatbot,
                  thought_block: mgr.Chatbot,
                  messages,
                  state
                  ):
            if state is None:
                semaphore = Semaphore(0)

                chat_messages = [{
                    ROLE: ASSISTANT,
                    CONTENT: [{
                        'text': "您好,请介绍你想要撰写的文书的信息"
                    }],
                }]
                writer_queue = Queue(100)
                agent, agent_list = build_agent_func(call_user_func=MyWebUI.call_user(semaphore, chat_messages),
                                                     writer_queue=writer_queue)
                chat_datas = ChatFuncData(
                    semaphore=semaphore,
                    chat_messages=chat_messages,
                    agent=agent,
                    agent_list=agent_list,
                    writer_queue=writer_queue,
                )
                state = str(time.time())
                state_dict[state] = chat_datas
            else:
                chat_datas = state_dict.get(state)
            semaphore = chat_datas.semaphore
            chat_messages = chat_datas.chat_messages
            agent = chat_datas.agent
            agent_list = chat_datas.agent_list
            messages.append({
                ROLE: USER,
                CONTENT: [{
                    'text': _input
                }],
            })
            chat_block.append([[{"text": _input}], None])
            chat_messages.append({
                ROLE: USER,
                CONTENT: [{
                    'text': _input
                }],
            })

            while True:
                thought_block.append([None, [None] * len(agent_list)])
                num_output_bubbles = 1
                yield (gr.update(interactive=False, value=None),
                       chat_block,
                       thought_block,
                       messages,
                       gr.update(interactive=False),
                       state)
                responses = []
                for responses in agent.run(messages):
                    if semaphore.acquire(blocking=False):
                        assert chat_messages[-1][ROLE] == ASSISTANT
                        chat_block.append([None, [{"text": chat_messages[-1][CONTENT][-1]['text']}]])
                        yield (gr.update(interactive=True, value=None),
                               chat_block,
                               thought_block,
                               messages,
                               gr.update(interactive=True),
                               state)
                        return
                        # print(responses)
                    if not responses:
                        continue

                    display_responses = convert_fncall_to_text(responses)
                    # print(display_responses)
                    if not display_responses:
                        continue
                    if display_responses[-1][CONTENT] is None:
                        continue
                    while len(display_responses) > num_output_bubbles:
                        # Create a new chat bubble
                        thought_block.append([None, [None] * len(agent_list)])
                        num_output_bubbles += 1
                    for i, rsp in enumerate(display_responses):
                        # print(rsp)
                        agent_index = agent_list.index(rsp[NAME])
                        thought_block[i - len(display_responses)][1][agent_index] = rsp[CONTENT]

                    yield (gr.update(interactive=False, value=None),
                           chat_block,
                           thought_block,
                           messages,
                           gr.update(interactive=False),
                           state)
                    # thought_block.append([None, [{"text": "思考中..."}]])
                messages.extend(responses)

        return _chat

    @staticmethod
    def call_user(semaphore, chat_messages):
        """
        将query加载到chat_messages里并设置信号量使chat函数进入另一个分支
        :return:/
        """

        def _call_user(query: str):
            chat_messages.append({
                ROLE: ASSISTANT,
                CONTENT: [{
                    'text': query
                }],
            })
            semaphore.release()

        return _call_user
