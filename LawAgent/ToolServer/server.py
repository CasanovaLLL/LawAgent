import os

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
import uvicorn
from LawAgent.Tools import CaseSearch, LawSearch, RelevantMarketSearch, PatternSearch, RecursiveClassifier

app = FastAPI()


class TextClassifierRequest(BaseModel):
    text: str
    classifier_name: Literal["General", "Monopoly"] = "Monopoly"


@app.post("/classifier")
async def call_classifier(request: TextClassifierRequest):
    if request.classifier_name == "General":
        classifier = RecursiveClassifier(
            tree_path=os.environ.get("GENERAL_CLASSIFIER_TREE_PATH",
                                     "data/preprocessed_data/general_classifier_tree.json"))
    else:
        classifier = RecursiveClassifier(
            tree_path=os.environ.get("MONOPOLY_CLASSIFIER_TREE_PATH",
                                     "data/preprocessed_data/monopoly_classifier_tree.json"))
    return {
        "class": classifier.call({"query": request.text})
    }


class SearchRequest(BaseModel):
    query: str
    labels: List[str] = []


@app.post("/search_laws")
def search_laws(request: SearchRequest):
    return {
        "laws": LawSearch().search(request.query, request.labels, True)
    }


@app.post("/search_case")
def search_case(request: SearchRequest):
    return {
        "case": CaseSearch().search(request.query, request.labels, True)
    }


@app.post("/relevant_market_case")
def search_relevant_market(request: SearchRequest):
    return {
        "case": RelevantMarketSearch().search(request.query, True)
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=10005)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=10005)
