import os
from qwen_agent.agents import GroupChat
from qwen_agent.gui import WebUI
from qwen_agent.llm.schema import Message
from LawAgent.Tools import *

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
CFGS = {
    'background': '一个生成和审查反垄断法诉讼文书的团队',
    'agents': [
        {
            'name': '咨询者',
            'description': '先来寻求代写反垄断相关文书的客户（这是一个真实用户）',
            'is_human': True
        },
        {
            'name': '文书模板确定者',  # 1. 文书模板确定
            'description': '根据案情选择合适的诉讼文书模板。',
            'instructions': '你负责确定是起诉书、上诉状还是再审申请书。然后告诉你的团队以便继续编写文书',
            'knowledge_files': [],
            'selected_tools': [PatternSearch()]
        },
        {
            'name': '反垄断法律专家',  # 3. 诉讼请求生成
            'description': '你根据',
            'instructions': '你是一位反垄断法律专家，根据案情、文书模板、指出对应的案由、并找出可能在编写中需要用到的相关的法律,交给文书编写者',
            'knowledge_files': [],
            'selected_tools': [RecursiveClassifier(os.environ.get("GENERAL_CLASSIFIER_TREE_PATH",
                                                                  "data/preprocessed_data/general_classifier_tree.json")),
                               LawSearch()
                               ]
        },
        {
            'name': '案例检索者',  # 6. 案例检索
            'description': '根据案由和案情匹配相关的反垄断案例。',
            'instructions': '你负责检索并匹配相关的反垄断案例与相关市场的数据。告诉你的团队',
            'knowledge_files': [],
            'selected_tools': [CaseSearch()]
        },
        {
            'name': '文书编写者',
            'description': '',
            'instructions': '你是一位反垄断方面的专家，你负责编写文书。'
                            '根据案情、文书模板确定者提供的模板、反垄断法律专家给出的案由和法条。'
                            '在文本编辑器中编写文书。如果遇到的任何问题可以再找反垄断法律专家或文书模板确定者核对',
            'knowledge_files': [],
            'selected_tools': [FileOperator()]
        },
        {
            'name': '审查者',
            'description': '审查文书是否符合最新的反垄断法民事诉讼司法解释。',
            'instructions': '你负责审查并确认文书符合最新司法解释。',
            'knowledge_files': [],
            'selected_tools': [LawSearch()]
        }
    ]
}


def app_tui():
    # Define a group chat agent from the CFGS
    bot = GroupChat(agents=CFGS, llm=llm_cfg, agent_selection_method="round_robin")
    # Chat
    messages = []
    while True:
        query = input('user question: ')
        messages.append(Message('user', query, name='咨询者'))
        response = []
        for response in bot.run(messages=messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    # Define a group chat agent from the CFGS
    bot = GroupChat(agents=CFGS, llm=llm_cfg)
    chatbot_config = {
        'user.name': '咨询者',
        'prompt.suggestions': [
        ],
        'verbose': True
    }

    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run()


if __name__ == '__main__':
    # app_tui()
    app_gui()
