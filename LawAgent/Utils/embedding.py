import os
from sentence_transformers import SentenceTransformer
import torch

__all__ = ["encode_long_text", 'embedding_model']
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "infgrad/stella-large-zh-v3-1792d")
WINDOWS_SIZE = int(os.getenv("EMBEDDING_WINDOW_SIZE", "512"))
embedding_model = SentenceTransformer(MODEL_PATH, device=device)


def encode_long_text(sentence: str):
    single_sentence = sentence.split("。")

    def _window_text_generate():
        target_sentence = []
        p = 0
        while p < len(single_sentence):
            target_sentence.append(single_sentence[p])
            string = "。".join(target_sentence)
            p += 1
            if len(string) < WINDOWS_SIZE / 1.3:
                continue
            if len(embedding_model.tokenize(string)) > WINDOWS_SIZE:
                if len(target_sentence) == 1:
                    yield string
                else:
                    target_sentence.pop(-1)
                    p -= 1
                    yield "。".join(target_sentence)

    target_string = [_ for _ in _window_text_generate()]
    return embedding_model.encode(target_string)


if __name__ == '__main__':
    import json
    from tqdm import tqdm

    o_data = []
    with open("new_ie.jsonl") as f:
        for line in tqdm(f, "Loading"):
            if line:
                o_data.append(json.loads(line))
    for ele in tqdm(o_data, "Encoding"):
        ele["embeddings"] = encode_long_text(ele["summary"])
        with open("new_ie_embedding.jsonl", "a") as f:
            f.write(json.dumps(ele, ensure_ascii=False) + "\n")
