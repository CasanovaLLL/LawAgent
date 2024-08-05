import time
import os
import requests
from requests import HTTPError
import concurrent.futures
from LawAgent.Utils import get_public_ip, get_computer_name_and_username, llm_response2json
import csv
from tqdm import tqdm
import json5
import json

__all__ = [
    "classify_text",
    "extract_law_from_text",
    "summarize_text",
]

URL = os.environ["DIFY_BASE_URL"] + "/workflows/run"
USERNAME = os.getenv("USERNAME", f"{get_public_ip()}-{'-'.join(get_computer_name_and_username())}")
DIFY_API_KEY_CLASSIFIER = os.environ["DIFY_API_KEY_CLASSIFIER"]
DIFY_API_KEY_LAW_EXTRACTION = os.environ["DIFY_API_KEY_LAW_EXTRACTION"]
DIFY_API_KEY_SUMMARIZATION = os.environ["DIFY_API_KEY_SUMMARIZATION"]


def classify_text(text: str) -> str:
    res_text = _request_dify_api(DIFY_API_KEY_CLASSIFIER, text)
    result = llm_response2json(res_text)
    if len(result):
        return result[0]
    return None


def extract_law_from_text(text: str) -> str:
    try:
        return _request_dify_api(DIFY_API_KEY_LAW_EXTRACTION, text).split("\n")
    except HTTPError:
        return None


def summarize_text(text: str) -> str:
    try:
        return _request_dify_api(DIFY_API_KEY_SUMMARIZATION, text)
    except HTTPError:
        return None


def _request_dify_api(api_key, body, **kwargs):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data_raw = {
        "input": body,
        "response_mode": "blocking",
        "user": USERNAME
    }
    response = _request_with_retry(URL, headers=headers, json=data_raw, **kwargs)
    if response:
        return response['data']['outputs']
    return None


def _request_with_retry(url, retries=3, *args, **kwargs):
    for _ in range(retries):
        try:
            response = requests.post(url, *args, **kwargs)
            response.raise_for_status()
            response = response.json()
            return response
        except requests.exceptions.RequestException as e:
            time.sleep(1)
    return None


def information_extraction(jsonl_path: str,
                           save_path: str,
                           fail_save_path: str,
                           num_workers: int = 4):
    """
    读取jsonl并处理，增加3个部分的内容，写回jsonl
    """

    def _process_one(_data):
        try:
            o_text = _data["o_text"]
            if o_text is None or o_text.strip() == "":
                return False, _data
            labels = classify_text(o_text)
            law_list = extract_law_from_text(o_text)
            summary = summarize_text(o_text)
            _data["labels"] = labels
            _data["laws"] = law_list
            _data["summary"] = summary
            if labels is None or law_list is None or summary is None:
                print("Error:", _data["filepath"])
                return False, _data
        finally:
            return True, _data

    with open(jsonl_path, 'r') as r:
        odata = [json5.loads(_) for _ in r.readlines()]
    with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        futures = []
        for ele in odata:
            futures.append(executor.submit(_process_one, ele))
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preprocessing"):
            flag, data = future.result()
            if flag:
                with open(save_path, 'a') as w:
                    w.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                with open(fail_save_path, 'a') as w:
                    w.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    information_extraction("/mnt/d/Codes/LawAgent/data/txt2csv", "ddata.jsonl")
