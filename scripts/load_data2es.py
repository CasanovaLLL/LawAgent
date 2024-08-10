from LawAgent.SearchEngine import LawDatabase

db = LawDatabase(search_with_embedding=False)
db.load_code("data/preprocessed_data/all_law_embedding.jsonl", "lawagent-code")
db.load_code("data/preprocessed_data/monopoly_relevent_clear_embedding.jsonl", "lawagent-code")
db.load_case("data/preprocessed_data/new_ie_embedding_with_otext.jsonl", "lawagent-case")
db.load_relevant_market('data/preprocessed_data/all_relevant_market.jsonl.with_embedding', 'lawagent-relevant')
