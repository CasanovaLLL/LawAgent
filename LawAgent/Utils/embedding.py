import os
from sentence_transformers import SentenceTransformer
import requests
import torch
from typing import Union, List

__all__ = ["encode_long_text", 'get_embedding_model']
EMBEDDING_URL = os.getenv("EMBEDDING_URL", None)


def get_embedding_model(model_name: str):
    if EMBEDDING_URL:
        print("使用在线Embedding服务")
        return OnlineEmbedding(model_name.split('/')[-1])
    return SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu",
                               trust_remote_code=True)


def encode_long_text(sentence: str):
    single_sentence = []
    tmp = sentence.split("\n\n")
    for _ in tmp:
        single_sentence.extend(_.split("。"))

    def _window_text_generate():
        target_sentence = []
        p = 0
        string = ""
        while p < len(single_sentence):
            target_sentence.append(single_sentence[p])
            string = "。".join(target_sentence)
            p += 1
            if len(string) < WINDOWS_SIZE / 1.3:
                continue
            if len(string) > WINDOWS_SIZE / 1.1:
                if len(target_sentence) == 1:
                    yield string
                    string = ""
                else:
                    target_sentence.pop(-1)
                    p -= 1
                    yield "。".join(target_sentence)
                    string = ""
                target_sentence = []
        if string.strip() != "":
            yield string

    target_string = [_ for _ in _window_text_generate()]
    stream = torch.Stream()
    with torch.cuda.stream(stream):
        return embedding_model.encode(target_string).tolist()


class OnlineEmbedding:
    def __init__(self, model_name="stella-large-zh-v3-1792d", base_url=EMBEDDING_URL):
        self.model = model_name
        self.base_url = base_url

    def encode(self, text: Union[str, List[str]]):
        response = requests.post(
            url=self.base_url,
            json={
                "model": self.model,
                "input": text
            }
        )
        response.raise_for_status()
        data = response.json()['data']
        data = sorted(data, key=lambda x: x["index"])
        embeddings = [_["embedding"] for _ in data]
        if isinstance(text, str):
            return embeddings[0]
        return embeddings


WINDOWS_SIZE = int(os.getenv("EMBEDDING_WINDOW_SIZE", "512"))
if not EMBEDDING_URL:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.getenv("STELLA_MODEL_PATH", "infgrad/stella-large-zh-v3-1792d")
    embedding_model = SentenceTransformer(MODEL_PATH, device=device, trust_remote_code=True)
else:
    embedding_model = OnlineEmbedding()

if __name__ == '__main__':
    import json
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    o_data = []
    with open("new_ie.jsonl") as f:
        for line in tqdm(f, "Loading"):
            if line:
                o_data.append(json.loads(line))


    def _encode(ele):
        ele["summary_embeddings"] = list(encode_long_text(ele["summary"]))
        ele["text_embeddings"] = list(encode_long_text(ele["o_text"]))
        return ele


    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for ele in o_data:
            futures.append(executor.submit(_encode, ele))
        for future in tqdm(as_completed(futures), "encoding", total=len(futures)):
            with open("new_ie_embedding_with_otext.jsonl", "a") as f:
                f.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
