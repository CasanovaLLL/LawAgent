import os
from sentence_transformers import SentenceTransformer
import torch

__all__ = ["encode_long_text", 'embedding_model']
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "infgrad/stella-large-zh-v3-1792d")
WINDOWS_SIZE = int(os.getenv("EMBEDDING_WINDOW_SIZE", "512"))
embedding_model = SentenceTransformer(MODEL_PATH, device=device, trust_remote_code=True)


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
