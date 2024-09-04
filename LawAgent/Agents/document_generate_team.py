import os
from qwen_agent.agents import GroupChat, UserAgent
from qwen_agent.gui import WebUI
from qwen_agent.llm.schema import Message
from LawAgent.Tools import *
from threading import Semaphore
from queue import Queue

# Define a configuration file for a multi-agent:
llm_cfg = {
    'model_type': 'oai',
    "model": "Qwen/Qwen2-72B-Instruct",
    'base_url': os.environ['LLM_BASE_URL'],
    'api_key': os.environ['LLM_API_KEY'],
    "generate_cfg": {
        "max_input_tokens": 20000
    }
}


def build_agent(call_user_func, writer_queue):
    team_notepad = NotePad()

    TEAM_CFGS = {
        'background': '一个生成和审查反垄断法诉讼文书的团队,团队成员各司其职确认模板、法条、类案、需求，整理后与用户交流。团队成员都不会废话，只会用简短的回答。',
        'agents': [
            {
                'name': '文书模板确定者',
                'description': '根据案情选择合适的诉讼文书模板。',
                'instructions': '你负责确定是起诉书、上诉状还是再审申请书，在记事本里添加模板。'
                                '并把应该向用户获取的资料添加到记事本里。以便你的团队以便继续准备编写文书的资料。\n'
                                '你只需要确定模板，不要做过多的分析\n'
                                '对话简短，把需要的信息都写在记事本里，其他发言在100字以内',
                'knowledge_files': [],
                'selected_tools': [PatternSearch(team_notepad), team_notepad]
            },
            {
                'name': '反垄断法律专家',
                'description': '根据案情找到相关的法律的人',
                'instructions': '你是一位反垄断法律专家，根据案情、文书模板、指出对应的案由、'
                                '并找出可能在编写中需要用到的相关的法律,添加到记事本里\n'
                                '你只需要找到信息并做简要分析，发言在100字以内\n'
                                '对话简短，把写文书需要的信息都写在记事本里',
                'knowledge_files': [],
                'selected_tools': [RecursiveClassifier(os.environ.get("GENERAL_CLASSIFIER_TREE_PATH",
                                                                      "data/preprocessed_data/monopoly_classifier_tree.json")),
                                   LawSearch(),
                                   team_notepad
                                   ]
            },
            {
                'name': '案例检索者',
                'description': '根据案由和案情匹配相关的反垄断案例的人',
                'instructions': '你负责检索并匹配相关的反垄断案例与相关市场的数据。添加到记事本里'
                                '对话简短，把需要的信息都写在记事本里，其他发言在100字以内',
                'knowledge_files': [],
                'selected_tools': [CaseSearch(), team_notepad]
            },
            {
                'name': '部门主管',
                'description': "部门主管让团队其他成员完成任务并与用户交流",
                'instructions': '根据现在的信息，确定是否需要向用户细化信息,还是直接交给文书编写者编写文书\n'
                                '对话简短，把需要的信息都写在记事本里，其他发言在100字以内\n'
                                '你的团队能看到你的所有发言，但用户只能看到communicate工具的发言，用户看不到附件，需要你把链接发给用户\n'
                                '除非你已经完成了所有工作，或者有信息需要用户提供，否则不要与用户沟通。结束你的发言即可\n',

                'knowledge_files': [],
                'selected_tools': [Communicate(call_user_func),
                                   team_notepad,
                                   ArticleWriter(team_notepad.absolute_path, llm=llm_cfg, queue=writer_queue)]
            }

        ]
    }
    agent = GroupChat(agents=TEAM_CFGS, description='一个生成和审查反垄断法诉讼文书的团队', llm=llm_cfg,
                      agent_selection_method='round_robin')
    agent_list = ['文书模板确定者', '反垄断法律专家', '案例检索者', '部门主管']
    return agent, agent_list
