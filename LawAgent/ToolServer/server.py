import os

from fastapi import FastAPI
from LawAgent.SearchEngine.elastic_search import LawDatabase
from pydantic import BaseModel
from typing import List, Literal
import uvicorn
from LawAgent.Tools.recursive_classifier import RecursiveClassifier

app = FastAPI()
db = LawDatabase()


class TextClassifierRequest(BaseModel):
    text: str
    classifier_name: Literal["General", "Monopoly"] = "Monopoly"


@app.post("/classifier")
async def call_classifier(request: TextClassifierRequest):
    print("Classify Target", request.text)
    if request.classifier_name == "General":
        classifier = RecursiveClassifier(
            tree_path=os.environ.get("GENERAL_CLASSIFIER_TREE_PATH",
                                     "data/preprocessed_data/general_classifier_tree.json"))
    else:
        classifier = RecursiveClassifier(
            tree_path=os.environ.get("MONOPOLY_CLASSIFIER_TREE_PATH",
                                     "data/preprocessed_data/monopoly_classifier_tree.json"))
    return {
        "class": classifier.call({"prompt": request.text})
    }


class LawSearchRequest(BaseModel):
    query: str
    labels: List[str]
    need_embedding: bool = False
    top_k: int = 5


@app.post("/search_laws")
async def search_laws(request: LawSearchRequest):
    embedding = None
    if request.need_embedding:
        raise NotImplementedError("need_embedding is not implemented yet")
    data = db.search_laws(request.query, request.labels, embedding, request.top_k)
    result = []
    if data:
        for ele in data:
            try:
                ele = ele["_source"]
                result.append(f"""
                {ele["depth"]}:{ele["code"]}
                """)
            except KeyError:
                pass
    return {
        "laws": result
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=10005)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=10005)
