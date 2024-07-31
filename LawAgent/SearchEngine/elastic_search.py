import os
import ssl
import warnings

from elasticsearch import Elasticsearch, helpers
from LawAgent.Utils import generate_timestamp
import json

_context = ssl.create_default_context(cafile=os.environ.get('ELASTICSEARCH_CA_PATH', 'http_ca.crt'))
"""
TODO
加载法典，加载类案
提供带label过滤器的相似度检索
"""


class Database:
    r"""
    所有的数据都应该包含两个字段：
    labels : list[str] 一个keyword 列表用来做筛选器
    embeddings: 密集向量列表 用来做相似度检索
    """

    def __init__(self, host='localhost', port=9200):
        auth = dict()
        try:
            auth['api_key'] = os.environ['ELASTICSEARCH_API_KEY']
        except KeyError:
            auth['basic_auth'] = os.environ.get(
                'ELASTICSEARCH_USERNAME', 'elastic'), os.environ['ELASTICSEARCH_PASSWORD']
        self.es = Elasticsearch(hosts=[{'host': host,
                                        'port': port,
                                        'scheme': 'https'}],
                                ssl_context=_context,
                                **auth
                                )

    def load_code(self, data_file_path, index_name=None):
        """
        数据加载到Elasticsearch中，并确保所有字段被正确索引。
        新的索引名称为 `code` 加上当前时间戳。

        """
        if not index_name:
            index_name = f'code{generate_timestamp()}'

        # 设置索引映射以指定数据类型
        body = {
            "mappings": {
                "properties": {
                    "code": {"type": "text"},
                    'labels': {"type": "keyword"},
                    "embeddings": {
                        "type": "nested",
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": 1792, "similarity": "cosine"}
                        }
                    }
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
                    ],
                    "filter": [
                        {"terms": {"labels": labels}}
                    ]
                }
            }
        }

        # 执行match查询
        match_results = self.es.search(index=index_name, body=match_query, size=top_k)

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
                        ],
                        "filter": [
                            {"terms": {"labels": labels}}
                        ]
                    }
                }
            }


            # 执行kNN查询
            knn_results = self.es.search(index=index_name, body=knn_query, size=top_k)

            return match_results, knn_results

        return match_results

    def _load_data(self, file_path, body, index_name):
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=body)

        # 定义一个生成器，用于从文件中读取数据并构建操作字典
        def data_generator():
            with open(file_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    yield {
                        "_index": index_name,
                        "_source": data
                    }

        try:
            # 使用helpers.bulk进行批量导入
            success, _ = helpers.bulk(self.es, data_generator())
            print(f'Successfully loaded {success} documents.')
        except helpers.BulkIndexError as e:
            print(f"Failed to load {len(e.errors)} documents.")
            for error in e.errors:
                print(f"Error indexing document: {error}")
