import os
from sentence_transformers import SentenceTransformer

__all__ = ["encode_long_text"]
MODEL_PATH = os.getenv("MODEL_PATH", "infgrad/stella-large-zh-v3-1792d")
WINDOWS_SIZE = int(os.getenv("EMBEDDING_WINDOW_SIZE", "512"))
model = SentenceTransformer(MODEL_PATH)


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
            if len(model.tokenize(string)) > WINDOWS_SIZE:
                if len(target_sentence) == 1:
                    yield string
                else:
                    target_sentence.pop(-1)
                    p -= 1
                    yield "。".join(target_sentence)

    target_string = [_ for _ in _window_text_generate()]
    return model.encode(target_string)
