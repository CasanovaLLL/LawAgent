import os
import ssl
import warnings

from elasticsearch import Elasticsearch, helpers
from LawAgent.Utils import generate_timestamp
import json
from tqdm import tqdm

try:
    _context = ssl.create_default_context(cafile=os.environ.get('ELASTICSEARCH_CA_PATH', 'http_ca.crt'))
except FileNotFoundError:
    pass
"""
TODO
加载法典，加载类案
提供带label过滤器的相似度检索
"""


class LawDatabase:
    r"""
    所有的数据都应该包含两个字段：
    labels : list[str] 一个keyword 列表用来做筛选器
    embeddings: 密集向量列表 用来做相似度检索
    """

    def __init__(self, host=os.environ.get("ES_HOST", "localhost"), port=int(os.environ.get("ES_PORT", 9200))):
        auth = dict()
        self.index_prefix = "lawagent-"
        self.law_index = os.environ.get("LAW_INDEX", "lawagent-law")

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

    def search_with_labels(self,
                           index_name: str,
                           query: str,
                           match_fields: list,
                           labels: list,
                           embedding=None,
                           top_k=10):
        """
        在指定的索引中搜索具有指定标签的文档，并返回最相关的文档。
        :param index_name: 索引名称
        :param query: 文本查询
        :param match_fields: 用来match的字段
        :param labels: 标签列表
        :param embedding: 向量用于kNN查询
        :param top_k: 返回的文档数量
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
        print(json.dumps(match_query, ensure_ascii=False))
        # 执行match查询
        match_results = self.es.search(index=index_name, body=match_query)

        # 第二步：如果提供了embedding，使用kNN查询和labels过滤器
        if embedding is not None:
            knn_query = {
                "query": {
                    "bool": {
                        "must": [
                                    {
                                        "nested": {
                                            "path": "embeddings",
                                            "query": {
                                                "knn": {
                                                    "embeddings.embedding": {
                                                        "vector": embedding,
                                                        "k": top_k
                                                    }
                                                }
                                            },
                                            "inner_hits": {}
                                        }
                                    }
                                ] + mast_query
                    }
                }
            }

            # 执行kNN查询
            knn_results = self.es.search(index=index_name, body=knn_query)

            return merge_and_rank(match_results["hits"]["hits"], knn_results["hits"]["hits"], top_k)

        return match_results["hits"]["hits"]

    def _load_data(self, file_path, body, index_name):
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=body)

        # 定义一个生成器，用于从文件中读取数据并构建操作字典
        def data_generator():
            with open(file_path, 'r') as file:
                for line in tqdm(file, desc="Upload"):
                    data = json.loads(line)
                    t = {
                        "_index": index_name,
                        "_source": data
                    }
                    print(json.dumps(t, ensure_ascii=False))
                    yield t
                    break

        try:
            # 使用helpers.bulk进行批量导入
            success, _ = helpers.bulk(self.es, data_generator())
            print(f'Successfully loaded {success} documents.')
        except helpers.BulkIndexError as e:
            print(f"Failed to load {len(e.errors)} documents.")
            for error in e.errors:
                print(f"Error indexing document: {error}")

    def search_laws(self,
                    query: str,
                    labels: list,
                    embedding=None,
                    top_k=5):
        result = self.search_with_labels(
            self.law_index,
            query,
            ["depth", "code"],
            labels,
            embedding,
            top_k
        )
        return result


def calculate_rrf_score(results, rank):
    """计算每个文档的RRF分数"""
    rrf_scores = {}
    for hit in results:
        doc_id = hit["_id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0
        if rank > 0:
            rrf_scores[doc_id] += 1 / rank
        rank += 1
    return rrf_scores


def merge_and_rank(match_results, knn_results, top_k):
    # 计算两个结果集的RRF分数
    match_rrf_scores = calculate_rrf_score(match_results, 1)
    knn_rrf_scores = calculate_rrf_score(knn_results, 1)

    # 合并结果并计算总分
    merged_results = []
    for doc_id, score in match_rrf_scores.items():
        total_score = score + knn_rrf_scores.get(doc_id, 0)
        merged_results.append({"_id": doc_id, "score": total_score})

    # 根据总分排序
    sorted_results = sorted(merged_results, key=lambda x: x["score"], reverse=True)

    # 返回前top_k个结果
    return [result["_id"] for result in sorted_results[:top_k]]
