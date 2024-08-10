import os
import ssl
import re
import warnings

from elasticsearch import Elasticsearch, helpers
from LawAgent.Utils import generate_timestamp
import json
from tqdm import tqdm
from typing import Union, List
from LawAgent.Utils import valid_label_match, check_labels
from LawAgent.Utils import SingletonMeta

try:
    _context = ssl.create_default_context(cafile=os.environ.get('ELASTICSEARCH_CA_PATH', 'http_ca.crt'))
except FileNotFoundError:
    pass


def _pure_return(func):
    def wrapper(*args, **kwargs):
        return [_["_source"] for _ in func(*args, **kwargs)]

    return wrapper


class LawDatabase(metaclass=SingletonMeta):
    r"""
    所有的数据都应该包含两个字段：
    labels : list[str] 一个keyword 列表用来做筛选器
    embeddings: 密集向量列表 用来做相似度检索
    """

    def __init__(self,
                 host=os.environ.get("ES_HOST", "localhost"),
                 port=int(os.environ.get("ES_PORT", 9200)),
                 search_with_embedding=True):
        auth = dict()
        self.search_with_embedding = search_with_embedding
        self.index_prefix = "lawagent-"
        self.law_index = os.environ.get("LAW_INDEX", "lawagent-code")
        self.case_index = os.environ.get("CASE_INDEX", "lawagent-case")

        self.relevant_market_index = os.environ.get("RELEVANT_INDEX", "lawagent-relevant")
        try:
            auth['api_key'] = os.environ['ELASTICSEARCH_API_KEY']
        except KeyError:
            auth['basic_auth'] = os.environ.get(
                'ES_USERNAME', 'elastic'), os.environ['ES_PASSWORD']
        self.es = Elasticsearch(hosts=[{'host': host,
                                        'port': port,
                                        'scheme': 'https'}],
                                # ssl_context=_context,

                                verify_certs=False,  # 忽略HTTPS证书检查
                                ssl_show_warn=False,
                                **auth
                                )
        if search_with_embedding:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.stella_embedding = SentenceTransformer(os.getenv("STELLA_MODEL_PATH",
                                                                  "infgrad/stella-large-zh-v3-1792d"),
                                                        device=device,
                                                        trust_remote_code=True)
            self.bge_m3 = SentenceTransformer(os.getenv("BGE_M3_PATH",
                                                        "BAAI/bge-m3"),
                                              device=device,
                                              trust_remote_code=True)

    def load_code(self, data_file_path, index_name=None):
        """
        数据加载到Elasticsearch中，并确保所有字段被正确索引。
        新的索引名称为 `code` 加上当前时间戳。

        """
        if not index_name:
            index_name = f'{self.index_prefix}code{generate_timestamp()}'

        # 设置索引映射以指定数据类型
        body = {
            "mappings": {
                "properties": {
                    "code": {"type": "text"},
                    'labels': {"type": "keyword"},
                    'depth': {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 1792, "similarity": "cosine"}
                }
            }}

        self._load_data(data_file_path, body, index_name)

    def load_case(self, data_file_path, index_name=None):
        """
        数据加载到Elasticsearch中，并确保所有字段被正确索引。
        新的索引名称为 `case` 加上当前时间戳。
        """
        if not index_name:
            index_name = f'{self.index_prefix}case{generate_timestamp()}'

        # 设置索引映射以指定数据类型
        body = {
            "mappings": {
                "properties": {
                    "o_text": {"type": "text"},  # 原始文本
                    'labels': {"type": "keyword"},  # 类别
                    'filepath': {"type": "text"},  # 文件路径
                    'text_embeddings': {
                        "type": "nested",
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": 1792, "similarity": "cosine"}
                        }
                    },
                    'summary': {"type": "text"},  # 摘要
                    'summary_embeddings': {
                        "type": "nested",
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": 1792, "similarity": "cosine"}
                        }
                    }
                }
            }}

        self._load_data(data_file_path, body, index_name, pre_process_case)

    def load_relevant_market(self, data_file_path, index_name=None):
        """
        数据加载到Elasticsearch中，并确保所有字段被正确索引。
        新的索引名称为 `relevant_market` 加上当前时间戳。嵌入模型为 BGE_M3
        """
        if not index_name:
            index_name = f'{self.index_prefix}relevant_market{generate_timestamp()}'
        body = {
            "mappings": {
                "properties": {
                    "embedding": {"type": "dense_vector", "dims": 1024, "similarity": "cosine"}
                }
            }
        }
        self._load_data(data_file_path, body, index_name)

    @_pure_return
    def search_with_labels(self,
                           index_name: str,
                           query: str,
                           match_fields: list,
                           labels: list,
                           embedding=None,
                           top_k=10,
                           embedding_field: Union[str, List[str]] = "embedding"):
        """
        在指定的索引中搜索具有指定标签的文档，并返回最相关的文档。
        :param index_name: 索引名称
        :param query: 文本查询
        :param match_fields: 用来match的字段
        :param labels: 标签列表
        :param embedding: 向量用于kNN查询
        :param top_k: 返回的文档数量
        :param embedding_field: 用来匹配向量的字段
        :return: 文档列表
        """
        mast_query = [{"term": {"labels": label}} for label in labels]
        # 第一步：使用match查询和labels过滤器
        match_query = {
            "query": {
                "bool": {
                    "must": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": match_fields
                                    }
                                }
                            ] + mast_query
                }
            }
        }
        # print(json.dumps(match_query, ensure_ascii=False))
        # 执行match查询
        match_results = self.es.search(index=index_name, body=match_query)

        # 第二步：如果提供了embedding，使用kNN查询
        if embedding is not None:
            if isinstance(embedding_field, str):
                mast_query.append({
                    "knn": {
                        "field": embedding_field,
                        "query_vector": embedding

                    }
                })
            else:
                assert isinstance(embedding_field, list)
                # 在多个embedding_fields里用nested搜索，采用“或”关系
                nested_queries = [{
                    "nested": {
                        "path": field,
                        "query": {
                            "knn": {
                                "field": f"{field}.embedding",
                                "query_vector": embedding,
                            }
                        },
                        "inner_hits": {}
                    }
                } for field in embedding_field]

                # 将所有nested查询包装在一个bool查询的'should'部分
                mast_query.append({
                    "bool": {
                        "should": nested_queries
                    }
                })

            knn_query = {
                "query": {
                    "bool": {
                        "must": mast_query  # 这里直接使用mast_query，因为它现在可能包含一个带有'should'的bool查询
                    }
                }
            }

            # 执行kNN查询
            knn_results = self.es.search(index=index_name, body=knn_query)

            return merge_and_rank(match_results["hits"]["hits"], knn_results["hits"]["hits"], top_k)

        return match_results["hits"]["hits"]

    def _load_data(self, file_path, body, index_name, pre_process=None):
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=body)

        # 定义一个生成器，用于从文件中读取数据并构建操作字典
        def data_generator():
            with open(file_path, 'r') as file:
                for line in tqdm(file, desc="Upload"):
                    data = json.loads(line)
                    if pre_process:
                        data = pre_process(data)
                    t = {
                        "_index": index_name,
                        "_source": data
                    }
                    yield t

        try:
            # 使用helpers.bulk进行批量导入
            success, _ = helpers.bulk(self.es, data_generator(), request_timeout=100)
            print(f'Successfully loaded {success} documents.')
        except helpers.BulkIndexError as e:
            print(f"Failed to load {len(e.errors)} documents.")
            for error in e.errors:
                print(f"Error indexing document: {error}")

    def search_laws(self,
                    query: str,
                    labels: list,
                    top_k=5):
        new_labels = check_labels(labels)
        if len(new_labels) == len(labels) and all(label in new_labels for label in labels):
            labels = new_labels
        else:
            labels = []
            query = '-'.join(labels) + ' ' + query
        embedding = self.stella_embedding.encode(query).tolist() if self.search_with_embedding else None
        result = self.search_with_labels(
            self.law_index,
            query,
            ["labels", "depth", "code"],
            labels,
            embedding,
            top_k
        )
        return result

    def search_cases(self,
                     query: str,
                     labels: list,
                     top_k=5):
        labels = check_labels(labels, only_tags=True)
        embedding = self.stella_embedding.encode(query).tolist() if self.search_with_embedding else None

        return self.search_with_labels(
            self.case_index,
            query,
            ["o_text", "summary"],
            labels,
            embedding,
            top_k,
            ["text_embeddings", "summary_embeddings"]
        )

    def search_relevant_market(self,
                               query: str,
                               top_k=5):
        embedding = self.bge_m3.encode(query).tolist() if self.search_with_embedding else None

        return self.search_with_labels(
            self.relevant_market_index,
            query,
            ["Case title", "Regulation",
             "Last decision date", "相关市场",
             "案件名称"],
            [],
            embedding,
            top_k
        )


def pre_process_case(data):
    new_labels = []
    for _ in data["use_labels"]:
        if _ := valid_label_match(_):
            new_labels.append(_)
    data["labels"] = new_labels
    data.pop("use_labels")
    data['summary_embeddings'] = [{"embedding": _} for _ in data['summary_embeddings']]
    data['text_embeddings'] = [{"embedding": _} for _ in data['text_embeddings']]
    return data


def merge_and_rank(match_results, knn_results, top_k):
    rrf_scores = {}
    results = {}
    K = 60

    def calculate_rrf_score(hhits):
        """计算每个文档的RRF分数"""

        for rank, hit in enumerate(hhits):
            rank = rank + 1
            _doc_id = hit["_id"]
            if _doc_id not in rrf_scores:
                rrf_scores[_doc_id] = 0
            if rank > 0:
                rrf_scores[_doc_id] += 1 / (rank + K)
            results[_doc_id] = hit

    # 计算两个结果集的RRF分数
    calculate_rrf_score(match_results)
    calculate_rrf_score(knn_results)

    merged_results = [{"_id": _id, "score": rrf_scores[_id]} for _id in rrf_scores.keys()]
    # 根据总分排序
    sorted_results = sorted(merged_results, key=lambda x: x["score"], reverse=True)

    # 返回前top_k个结果
    return [results[result["_id"]] for result in sorted_results[:top_k]]


if __name__ == '__main__':
    es = LawDatabase()
    print(es.search_cases("公司", []))
