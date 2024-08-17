r"""
给结果打分
计算方式为计算 BGE-m3的相似度分数与关键词覆盖率
为6：4
"""
K = 0.6

import pandas as pd
from FlagEmbedding import BGEM3FlagModel

#
# sentences_1 = ["What is BGE M3?", "Defination of BM25"]
# sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
#                "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]
#
# sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
df = pd.read_csv(r'反垄断-问答20题-命中率.csv')


def count_hit(use_col):
    def _count_hit(row):
        keywords = row['关键词'].split(';')
        keywords = [keyword.strip() for keyword in keywords]
        hit_count = 0
        for k in keywords:
            if k in use_col:
                hit_count += 1
        return hit_count / len(keywords)

    return _count_hit


def cal_beg_m3_scroe(df, use_col):
    text_pairs_list = []
    for idx, row in df.iterrows():
        ori_answer = row['A']
        llm_answer = row[use_col]
        text_pairs = [ori_answer, llm_answer]
        text_pairs_list.append(text_pairs)
    print(text_pairs_list)
    score_dict = model.compute_score(text_pairs_list,
                                     max_passage_length=1024,
                                     weights_for_different_modes=[0.4, 0.2, 0.4])
    score_df = pd.DataFrame(score_dict)
    # score_df['No']=list(range(1,len(df)+1))
    return score_df


use_col = 'lawAgent'
df['lawAgent_hit_ratio'] = df.apply(count_hit('lawAgent'), axis=1)
law_score_df = cal_beg_m3_scroe(df, use_col)
# 使用rename方法添加前缀
law_score_df.rename(columns=lambda x: f'{use_col}{x}', inplace=True)
#
#
# use_col = 'Qwen/Qwen2-72B-Instruct_ans'
# qwen_score_df = cal_beg_m3_scroe(df, use_col)
# # 使用rename方法添加前缀
# qwen_score_df.rename(columns=lambda x: f'{use_col}{x}', inplace=True)
# print(qwen_score_df)
#
# use_col = 'deepseek-ai/DeepSeek-V2-Chat_ans'
# deepseek_score_df = cal_beg_m3_scroe(df, use_col)
# # 使用rename方法添加前缀
# deepseek_score_df.rename(columns=lambda x: f'{use_col}{x}', inplace=True)
# print(deepseek_score_df)
# final_save_df = pd.concat([qwen_score_df, deepseek_score_df], axis=1)
# final_save_df['No'] = list(range(1, len(final_save_df) + 1))
# df = df.merge(final_save_df, on='No', how='left')

target_cols = ['lawAgent', 'Qwen/Qwen2-72B-Instruct_ans', 'deepseek-ai/DeepSeek-V2-Chat_ans']


def count_finally_score(row):
    for col in target_cols:
        row[f'{col}_score'] = K * row[f'{col}_hit_ratio'] + (1 - K) * row[f'{col}colbert+sparse+dense']
    return row


score_df = df.apply(count_finally_score, axis=1)

df.to_csv("反垄断-问答20题-命中率-bge-m3.csv", index=0)
