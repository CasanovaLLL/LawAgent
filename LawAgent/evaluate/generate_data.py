import os
import openai
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
