r"""
    解析 https://github.com/ImCa0/just-laws/ 下的法条为jsonl
    读取每一个md文件
    提取出 label list，为这一法条的层级，所属部门法
    最终为一个jsonl文件，每行一个法条
    同时存在精确到条、款、项的元数据
    单元数据样例
    {
  "labels":[
    "中华人民共和国反垄断法",
    "第三章 滥用市场支配地位",
    "第二十二条",
    "第三款",
    "中华人民共和国反垄断法-第三章-第二十二条-第三款"
  ],
  "depth":"中华人民共和国反垄断法-第三章-第二十二条-第三款",
  "text":"（二）没有正当理由，以低于成本的价格销售商品；"
}
"""
import markdown_to_json
import os
from tqdm import tqdm
import json
import re
from LawAgent.Utils import unique_with_order_list_comprehension
import cn2an
import pandas as pd


def walk_dir(dir_path: str):
    result = dict()
    for root, dirs, files in tqdm(os.walk(dir_path)):
        files = [_ for _ in files if _.endswith('.md')]
        if "amendment" in root:
            continue
        if "README.md" not in files:
            print("README.md not found in {}".format(root))
            continue
        with open(os.path.join(root, "README.md"), "r", encoding="utf-8") as f:
            root_data = markdown_to_json.dictify(f.read())

        title = list(root_data.keys())[0]
        if not isinstance(root_data[title], list):
            root_data[title] = {title: root_data[title]}
        files.remove("README.md")
        for file in files:
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                data = markdown_to_json.dictify(f.read())
                root_data[title].update(data)
        result[title] = root_data
    return result


def parse_law(data: dict):
    result = []

    def _dfs(node: dict, _label_list: list):
        for k, v in node.items():
            labels = _label_list + ([k] if k != '```' else [])
            if isinstance(v, dict):
                _dfs(v, labels)
            else:
                newv = _parse_article_text_v2(v)
                if len(newv) > 0:
                    _dfs(newv, labels)
                    continue
                numbers = [labels[0]]
                for label in labels:
                    numbers.extend(parse_label_number(label))
                depth = '-'.join(numbers)
                labels.append(depth)
                labels = unique_with_order_list_comprehension(labels)
                labels = [_.replace('\u3000', ' ') for _ in labels]
                result.append({
                    "labels": labels,
                    "depth": "-".join(numbers),
                    "text": v
                })

    _dfs(data, [])
    return result


def parse_label_number(text: str):
    pattern = r"([第][一二三四五六七八九十百千万亿]+[章节篇款条编项]|(?<![第][一二三四五六七八九十百千万亿][章节篇款条编项][\s\u3000])附则)"
    matches = re.finditer(pattern, text)
    result = []
    for match in matches:
        result.append(match.group(1))
    return result


def _parse_article_text(text: str):
    """
    条
    :param text:
    :return:
    """
    pattern = r"\*\*\s*([^\s\*]+[^\*]*)(?!\*\*)*\*\*"
    matches = re.finditer(pattern, text)

    result = {}
    start_index = 0
    key = None
    for match in matches:
        new_key = match.group(1).strip()
        end_index = match.start()  # 当前键的开始位置即前一个值的结束位置

        value = text[start_index:end_index].strip()
        if len(value) > 0 and key:
            result[key] = _parse_clause_text(value)
        key = new_key
        start_index = match.end()  # 更新下一个值的开始位置

    # 添加最后一个值
    last_value = text[start_index:].strip()
    if last_value and key:
        result[key] = last_value
    return result

def _parse_article_text_v2(text: str):
    """
    条
    :param text:
    :return:
    """
    pattern = r"\s+(第[一二三四五六七八九十百千万亿]+条)\s+"
    matches = re.finditer(pattern, text)

    result = {}
    start_index = 0
    key = None
    for match in matches:
        new_key = match.group(1).strip()
        end_index = match.start()  # 当前键的开始位置即前一个值的结束位置

        value = text[start_index:end_index].strip()
        if len(value) > 0 and key:
            result[key] = value
        key = new_key
        start_index = match.end()  # 更新下一个值的开始位置

    # 添加最后一个值
    last_value = text[start_index:].strip()
    if last_value and key:
        result[key] = last_value
    return result


def _parse_clause_text(text: str):
    """
    款
    :param text:
    :return:
    """

    texts = [_ for _ in text.split('\n') if len(_.strip()) > 0]

    result = {
        "```": text.strip()
    }
    if len(texts) == 1:
        return result
    counter = 0
    clause_texts = []
    name = None
    for text in texts:
        item = _parse_item_text(text)
        if len(item) > 0:
            result[name].update(item)
        else:
            counter += 1
            name = "第" + cn2an.an2cn(counter, "low") + "款"
            result[name] = dict()
            clause_texts = []
        clause_texts.append(text)
        result[name]["```"] = '\n\n'.join(clause_texts)
    if len(result) == 2:
        result = result["第一款"]
    return result


def _parse_item_text(text: str):
    """
    项
    :param text:
    :return:
    """
    pattern = r'\s*（([零一二三四五六七八九十百千万亿]+)）\s*([^\n]+)'
    match = re.match(pattern, text)
    if not match:
        return {}
    name = '第' + match.group(1) + '项'
    value = match.group(2)

    return {name: value}


def _insert_labels(jsonl_path, xlsx_path):
    df = pd.read_excel(xlsx_path, dtype=str)
    col = df.columns
    with open(jsonl_path, "r", encoding="utf-8") as f:
        laws = f.readlines()
        laws = [json.loads(_, strict=False) for _ in tqdm(laws, "Loading ")]
    for law in tqdm(laws, desc="Checking "):
        for row in df.values:
            labels = [row[0], row[1]]
            pos = '-'.join(['第' + _ for _ in row[-1].split("第") if _.strip() != ""])
            # print(col[-1],pos)
            if col[-1] in law["depth"] and pos in law["depth"]:
                law["labels"].extend(labels)
                print(row)

    with open(jsonl_path + '.tmp', "w", encoding="utf-8") as f:
        for law in laws:
            f.write(json.dumps(law, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    # print(json.dumps(walk_dir("./docs"), ensure_ascii=False, indent=4))
    # from pprint import pprint
    #
    # data = walk_dir("./docs")
    # data = parse_law(data)
    # with open("./docs/all_law.jsonl", "w", encoding="utf-8") as f:
    #     for item in data:
    #         f.write(json.dumps(item, ensure_ascii=False) + "\n")
    _insert_labels("./all_law.jsonl", "../副本分类--法条 对应.xlsx")
