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
from typing import Literal
import json
import jsonlines

__all__ = [
    "classify_text",
    "extract_law_from_text",
    "summarize_text",
]
try:
    URL = os.environ["DIFY_BASE_URL"] + "/workflows/run"
    USERNAME = os.getenv("USERNAME", f"{get_public_ip()}-{'-'.join(get_computer_name_and_username())}")
    DIFY_API_KEY_CLASSIFIER = os.environ["DIFY_API_KEY_CLASSIFIER"]
    DIFY_API_KEY_LAW_EXTRACTION = os.environ["DIFY_API_KEY_LAW_EXTRACTION"]
except KeyError:
    pass


def classify_text(text: str) -> str:
    if res_text := _request_dify_api(DIFY_API_KEY_CLASSIFIER, text):
        res_text = res_text["output"]
    else:
        return None
    try:
        result = llm_response2json(res_text)
        if len(result):
            return result[0]
    finally:
        return res_text


def extract_law_from_text(text: str) -> str:
    try:
        laws = _request_dify_api(DIFY_API_KEY_LAW_EXTRACTION, text)["output"].split("\n")
        laws = [_ for _ in laws if "没有找到" in _ or _.strip() != ""]
        return laws
    except HTTPError:
        return None


def summarize_text(text: str, text_type: Literal["penalty", "judgment", "notice"]) -> str:
    """

    :param text:
    :param text_type: 分别代表行政处罚、判决或裁定书、经营者集中公告
    :return:
    """
    target_key = {
        "penalty": "DIFY_API_KEY_SUMMARIZATION_PENALTY",
        "judgment": "DIFY_API_KEY_SUMMARIZATION_JUDGMENT",
        "notice": "DIFY_API_KEY_SUMMARIZATION_NOTICE"
    }
    try:
        if response := _request_dify_api(os.environ[target_key[text_type]], text):
            return response["output"]
        return None
    except HTTPError:
        return None


def _request_dify_api(api_key, text, **kwargs):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data_raw = {
        "inputs": {"query": text},
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
            o_path = _data["filepath"]
            text_type = None
            if "行政处罚决定书" in o_path:
                text_type = "penalty"
            elif "裁定书" in o_path:
                text_type = "judgment"
            elif "经营者集中" in o_path:
                text_type = "notice"

            labels = classify_text(o_text)
            _data["labels"] = labels

            law_list = extract_law_from_text(o_text)
            _data["laws"] = law_list

            summary = summarize_text(o_text, text_type)
            _data["summary"] = summary

            if labels is None or law_list is None or summary is None:
                print("Error:", _data["filepath"])
                return False, _data
            return True, _data
        except:
            return False, _data

    with open(jsonl_path, 'r') as r:
        odata = [json5.loads(_) for _ in tqdm(r.readlines()[:1], desc="Loading")]
    print("Start processing...")
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


def labels_extraction():
    with open('ie.jsonl', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            json_data = json.loads(line)
            labels_str = json_data['labels']
            labels_str = labels_str.replace('```json\n', '').replace('\n```', '')
            labels_json_data = json.loads(labels_str)
            labels_key_list = []
            for k, v in labels_json_data.items():
                if k == '分类':
                    try:
                        v_list = eval(str(v))
                        for v_item in v_list:
                            temp_key_list = v_item.split("-")
                            labels_key_list += temp_key_list
                    except BaseException as e:
                        print(str(e))
                        print(labels_json_data)
                elif k == 'categories':
                    try:
                        v_list = eval(str(v))
                        for v_item in v_list:
                            temp_key_list = v_item.split("-")
                            labels_key_list += temp_key_list
                    except BaseException as e:
                        print(str(e))
                        print(labels_json_data)
                elif v != False and v != True and len(str(v)) > 2:
                    temp_key_list = k.split("-")
                    labels_key_list += temp_key_list
                if v == True:
                    temp_key_list = k.split("-")
                    labels_key_list += temp_key_list
            labels_key_list = list(set(labels_key_list))
            json_data['use_labels'] = labels_key_list
            if len(labels_key_list) > 0:
                with jsonlines.open('new_ie.jsonl', mode='a') as writer:
                    writer.write(json_data)
            else:
                with jsonlines.open('new_ie_fail.jsonl', mode='a') as writer:
                    writer.write(json_data)


if __name__ == '__main__':
    labels_extraction()
