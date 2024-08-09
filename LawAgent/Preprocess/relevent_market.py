import os

from sentence_transformers import SentenceTransformer
import pandas as pd
import json


def load_and_process_excel(file_path):
    """
    加载一个Excel文件的所有sheet，并进行预处理。

    原始表格有以下几列:
    - 序号
    - 案件名称
    - 公示时间
    - 相关市场
    - 类型 (如果不存在，则默认为'允许集中')

    处理步骤:
    - 删除'序号'列
    - 如果'类型'列不存在，则添加并填充为'允许集中'
    - 将所有sheet合并为一个DataFrame

    参数:
    - file_path: Excel文件的路径

    返回:
    - 合并后的DataFrame
    """

    # 读取Excel文件中的所有sheet
    sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)

    processed_dfs = []
    for sheet_name, sheet_df in sheets.items():
        # 删除'序号'列
        if '序号' in sheet_df.columns:
            sheet_df.drop(columns=['序号'], inplace=True)

        # 如果'类型'列不存在，则添加并填充为'允许集中'
        if '类型' not in sheet_df.columns:
            sheet_df['类型'] = '允许集中'

        processed_dfs.append(sheet_df)

    # 合并所有处理过的DataFrame
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    return combined_df


def encode(file_path):
    target_cal = ["Economic activities", "相关市场"]
    embedding_model = SentenceTransformer(os.getenv("BCE_M3_PATH", "BAAI/bge-m3"))
    with open(file_path, 'r', encoding='utf-8') as f:
        datas = f.readlines()
    datas = [json.loads(data) for data in datas]
    target_text = []
    for data in datas:
        for target in target_cal:
            if target in data:
                target_text.append(data[target])
    embeddings = embedding_model.encode(target_text).tolist()
    for data, embedding in zip(datas, embeddings):
        data["embedding"] = embedding
    with open("embeddings_" + file_path, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
