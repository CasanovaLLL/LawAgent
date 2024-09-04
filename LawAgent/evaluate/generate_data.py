import os
import openai
import zhipuai
from LawAgent.Utils.dify_api import DifyAPI


def generate_dify_data(query):
    client = DifyAPI(os.getenv("DIFY_CHAT_API_KEY"), os.getenv("DIFY_CHAT_API_URL"))
    return client.conversation(query, inputs={"input_function": "情景问答"})


def generate_oai_data(model_name: str):
    def _gen(query):
        format_model_name = f"{model_name}".upper()
        format_model_name = format_model_name.replace("-", "_")
        client = openai.OpenAI(api_key=os.getenv(f"{format_model_name.upper()}_API_KEY", os.getenv("LLM_API_KEY", "")),
                               base_url=os.getenv(f"{format_model_name.upper()}_API_URL", os.getenv("LLM_BASE_URL")))
        return client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": query},
            ],
            stream=False
        ).choices[0].message.content

    return _gen


def generate_zhipuai_data(model_name: str):
    def _gen(query):
        client = zhipuai.ZhipuAI(api_key=os.getenv('ZHIPU_APIKEY'))
        response = client.chat.completions.create(
            model=model_name,  # 填写需要调用的模型名称
            messages=[
                {"role": "user",
                 "content": query},
            ],
            tools=[{"type": "web_search", "web_search": {"search_result": True}}],
            stream=True,
            temperature=0.95,
            max_tokens=1024,
        )
        answer_parts = []
        for chunk in response:
            if content := chunk.choices[0].delta.content:
                answer_parts.append(content)
        response_str = "".join(answer_parts)
        return response_str

    return _gen


def generate_data_from_excel(excel_path):
    import pandas as pd
    df = pd.read_excel(excel_path)
    df["Q"] = df["Q"].apply(lambda x: x.strip())
    dictionary = df.set_index('Q')['A'].to_dict()

    def _gen(query):
        query = query.strip()
        return dictionary.get(query)

    return _gen


def generate_data_from_tongyi_farui_api():
    """
    7毛一次也太贵了！
    TODO
    """
    from alibabacloud_tea_openapi import models as open_api_models
    from alibabacloud_farui20240628.client import Client as Client
    from alibabacloud_farui20240628 import models as models

    access_key_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
    access_key_secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    config = open_api_models.Config(
        # 您的AccessKey ID,
        access_key_id=access_key_id,
        # 您的AccessKey Secret,
        access_key_secret=access_key_secret
    )
    # 访问的域名
    config.endpoint = 'farui.cn-beijing.aliyuncs.com'

    client = Client(config)
    request = models.RunLegalAdviceConsultationRequest()
    # 该参数值为假设值，请您根据实际情况进行填写
    request.app_id = "LawAgent"

    # 该参数值为假设值，请您根据实际情况进行填写
    request.stream = False

    # 该参数值为假设值，请您根据实际情况进行填写
    request.workspace_id = "your_value"
