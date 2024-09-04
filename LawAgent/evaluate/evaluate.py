import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from LawAgent.Utils.embedding import get_embedding_model
from LawAgent.Utils.utils import sanitize_filename

SIMILARITY_MODEL = "bge-m3"
_model = get_embedding_model(SIMILARITY_MODEL)
KEYWORDS_PROPORTION = 0.5


class EvaluateDataset:
    def __init__(self, data_path: str = None):

        self.data = pd.read_excel(data_path)
        self.gold_answer = self.data["A"].tolist()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_model.encode, text) for text in self.data["A"]]
            ans_embedding = [future.result() for future in futures]
            self.gold_embedding = ans_embedding

        self.keyword_list = []
        for ele in self.data["Keywords"].tolist():
            self.keyword_list.append([x.strip() for x in ele.split(";")])
        self.gold_score = []
        for ans, keywords in zip(self.gold_answer, self.keyword_list):
            score = KEYWORDS_PROPORTION * sum([_ in ans for _ in keywords]) / len(keywords) + (1 - KEYWORDS_PROPORTION)
            self.gold_score.append(score)
        print("load evaluate data done")

    def evaluate(self, llm_generate_func, llm_name: str):
        llm_name = sanitize_filename(llm_name)
        print("start evaluate")
        os.makedirs("data/evaluate/llmOutput/", exist_ok=True)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for q in self.data["Q"].tolist():
                futures.append(executor.submit(llm_generate_func, q))
            llm_response = [future.result() for future in tqdm(futures, desc=f"Get `{llm_name}` Response")]
            store_pd = pd.DataFrame({llm_name: llm_response})
            store_pd.to_csv(f"data/evaluate/llmOutput/{llm_name}.csv", index=False)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_model.encode, text) for text in llm_response]
            ans_embeddings = [future.result() for future in tqdm(futures, desc=f"Get `{llm_name}` Embedding")]
            cos_sim_score_list = []
            for gold_embedding, ans_embedding in zip(self.gold_embedding, ans_embeddings):
                gold_embedding = np.array(gold_embedding)
                ans_embedding = np.array(ans_embedding)
                score = np.dot(gold_embedding, ans_embedding)
                cos_sim_score_list.append(score)
            store_pd["cos_sim_score"] = cos_sim_score_list
        total_score = []
        keyword_score_list = []
        for response, keywords, cos_sim_score in zip(llm_response, self.keyword_list, cos_sim_score_list):
            keyword_score = sum([_ in response for _ in keywords]) / len(keywords)
            score = KEYWORDS_PROPORTION * keyword_score + (
                    1 - KEYWORDS_PROPORTION) * cos_sim_score
            total_score.append(score)
            keyword_score_list.append(keyword_score)
        store_pd["keyword_score"] = keyword_score_list
        store_pd["total_score"] = total_score
        store_pd.to_csv(f"data/evaluate/llmOutput/{llm_name}.csv", index=False)
        print("evaluate done")
        return store_pd


if __name__ == '__main__':
    evaluate_dataset = EvaluateDataset('data/evaluate/反垄断QA20.xlsx')
    from LawAgent.evaluate.generate_data import generate_dify_data, generate_oai_data, generate_zhipuai_data,generate_data_from_excel

    # evaluate_dataset.evaluate(generate_dify_data, "dify")
    # evaluate_dataset.evaluate(generate_oai_data("meta-llama/Meta-Llama-3.1-405B-Instruct"), "meta-llama/Meta-Llama-3.1-405B-Instruct")
    # generate_zhipuai_data('glm-4-plus')
    evaluate_dataset.evaluate(generate_data_from_excel('data/evaluate/llmOutput/tongyiFarui.xlsx'),
                              "farui")