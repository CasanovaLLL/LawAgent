import requests
import socket
import getpass
from datetime import datetime
import os
import csv
from tqdm import tqdm
import json5
import json
import re
import ast
from .constant import *
import editdistance

__all__ = [
    'generate_timestamp',
    'dir_txt2csv',
    'get_public_ip',
    'get_computer_name_and_username',
    'llm_response2json',
    'dir_txt2jsonl',
    'build_username',
    'unique_with_order_list_comprehension',
    'VALID_TAGS',
    'LAW_NAMES',
    'valid_label_match',
    'law_name_match',
    'find_nearest_text',
    'check_labels',
    'SingletonMeta'
]


def generate_timestamp():
    """
    生成当前时间的yyyyMMddhhmmss格式的时间戳字符串。
    """
    now = datetime.now()  # 获取当前时间
    formatted_time = now.strftime('%Y%m%d%H%M%S')  # 格式化时间字符串
    return formatted_time


def dir_txt2csv(work_dir, save_dir):
    """
    递归处理文件夹下的txt文件，将目录下的所有txt文件转换为csv的一行，其余文件忽略
    在save_dir文件夹下保存为同样的文件层级结构
    :param work_dir: 源目录路径
    :param save_dir: 目标目录路径
    :return: None
    """

    # 确保目标目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历工作目录
    for root, dirs, files in os.walk(work_dir):
        # 计算相对于工作目录的路径
        relative_path = os.path.relpath(root, work_dir)
        # 如果相对路径不是当前目录，则创建对应的子目录
        if relative_path != '.':
            target_subdir = os.path.join(save_dir, relative_path)
            os.makedirs(target_subdir, exist_ok=True)

        # 初始化CSV文件路径
        csv_file_path = os.path.join(save_dir, relative_path, f"{os.path.basename(root)}.csv")

        # 打开或创建CSV文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            # 写入表头
            csv_writer.writerow(['target'])

            # 使用tqdm显示进度条
            for file in tqdm(files, desc=f"Processing {relative_path}", unit="file"):
                if file.endswith('.txt'):
                    # 构建源文件的完整路径
                    txt_file_path = os.path.join(root, file)

                    # 读取txt文件内容
                    with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
                        content = txt_file.read().strip()

                    # 写入内容
                    csv_writer.writerow([content])


def dir_txt2jsonl(work_dir, save_path):
    """
    递归处理文件夹下的txt文件，将目录下的所有txt文件转换为csv的一行，其余文件忽略
    在save_path中保存 包括键 relpath表示路径 和 o_text表示原始数据
    :param work_dir: 源目录路径
    :param save_path: 目标目录路径
    :return: None
    """
    all_data = []
    # 遍历工作目录
    for root, dirs, files in os.walk(work_dir):
        # 计算相对于工作目录的路径
        relative_path = os.path.relpath(root, work_dir)

        # 使用tqdm显示进度条
        for file in tqdm(files, desc=f"Processing {relative_path}", unit="file"):
            if file.endswith('.txt'):
                # 构建源文件的完整路径
                txt_file_path = os.path.join(root, file)

                # 读取txt文件内容
                with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
                    content = txt_file.read().strip()
                    all_data.append({
                        "filepath": txt_file_path,
                        "o_text": content
                    })
    with open(save_path, 'w', encoding='utf-8') as jsonl_file:
        for data in tqdm(all_data, desc="Writing..."):
            jsonl_file.write(json.dumps(data, ensure_ascii=False) + '\n')


def get_public_ip():
    """
    获取本机的公网IP地址。获取不到则返回0.0.0.0
    :return: 公网IP地址字符串
    """
    try:
        # 尝试从不同的服务获取公网IP
        response = requests.get("https://ident.me", timeout=5)
        if response.status_code == 200:
            return response.text.strip()

        response = requests.get("http://txt.go.sohu.com/ip/soip", timeout=5)
        if response.status_code == 200:
            return response.text.strip()

        response = requests.get("https://myip.ipip.net", timeout=5)
        if response.status_code == 200:
            return response.text.split("：")[1].strip()
    finally:
        pass

    return "0.0.0.0"


def get_computer_name_and_username():
    """
    获取本机计算机名和当前用户名。
    :return: (计算机名, 用户名) 的元组
    """
    computer_name = socket.gethostname()
    username = getpass.getuser()
    return computer_name, username


def llm_response2json(text: str):
    """
    从文本中抽取json格式的内容
    正则匹配json或python代码块或者直接匹配
    先尝试用json5.loads(),再尝试用python.eval()
    返回所有的块解析后的内容
    若没有块，则尝试直接用json5.loads()整个块
    :param text:
    :return: List[Union[Dict,List]]
    """
    pattern = r'```(?:json|python)?\n(.*?)\n```'
    # 查找所有匹配项
    matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)

    # 解析所有匹配的代码块
    parsed_blocks = []
    for match in matches:
        try:
            # 尝试用json5解析
            parsed = json5.loads(match)
            parsed_blocks.append(parsed)
        except ValueError:
            try:
                # 尝试用ast.literal_eval解析
                parsed = ast.literal_eval(match)
                parsed_blocks.append(parsed)
            except (ValueError, SyntaxError):
                # 如果无法解析，则跳过
                continue

    # 如果没有匹配到任何块，则尝试直接解析整个文本
    if not parsed_blocks:
        try:
            parsed = json5.loads(text)
            parsed_blocks.append(parsed)
        except ValueError:
            pass

    return parsed_blocks


def build_username():
    return os.getenv("USERNAME", f"{get_public_ip()}-{'-'.join(get_computer_name_and_username())}")


def find_nearest_text(text_list: list, query, threshold=1):
    min_distance = 1
    _result = None
    for _ in text_list:
        distance = editdistance.eval(query, _) / max(len(query), len(_))
        if distance < min_distance:
            min_distance = distance
            _result = _
    if min_distance < threshold:
        return _result
    return None


def check_labels(labels: list, only_tags: bool = False):
    index_pattern = r"([第][一二三四五六七八九十百千万亿0-9]+[章节篇款条编项])"
    title_pattern = r"(中国[\u4e00-\u9fa5]+法|中华人民共和国[\u4e00-\u9fa5]+法|[\u4e00-\u9fa5]+法)"
    labels = list(set(labels))
    for i, label in enumerate(labels):
        if not only_tags and (re.match(index_pattern, label) or re.match(title_pattern, label)):
            continue
        label = valid_label_match(label)
        labels[i] = label
    labels = list(set(labels))
    try:
        labels.remove(None)
    except ValueError:
        pass
    return labels


def law_name_match(query):
    pattern = r'(?:中[\u4e00-\u9fa5]?国)?([\u4e00-\u9fa5]+)法'
    new_law_list = [re.match(pattern, _).group(1) for _ in LAW_NAMES]

    if res := re.match(pattern, query):
        res.group(1)
        return find_nearest_text(new_law_list, res.group(1), threshold=0.5)
    return None


def valid_label_match(query):
    return find_nearest_text(VALID_TAGS, query, threshold=0.5)


def unique_with_order_list_comprehension(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


class SingletonMeta(type):
    """
    元类实现单例模式
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
