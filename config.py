from dataclasses import dataclass

@dataclass
class Config:
    neo4j_url:  str = "bolt://127.0.0.1:7415"
    neo4j_user: str = "neo4j"
    neo4j_pass: str = "medicalgraphrag"
    
    base_url:   str = "http://localhost:7414"
    model:      str = "qwen2.5:32k"
    emb_model:  str = "cwchang/jina-embeddings-v2-base-zh"
    
config = Config()