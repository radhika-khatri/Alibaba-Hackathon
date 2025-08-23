import os
from pinecone import Pinecone, ServerlessSpec


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "kb-index")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")


# Ensure index exists
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
    name=INDEX_NAME,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )

index = pc.Index(INDEX_NAME)

def upsert_vectors(items: list[dict]):
    # items: [{id, values(list[float]), metadata(dict)}]
    index.upsert(vectors=items)

def query_vector(vec: list[float], top_k: int = 4):
    return index.query(vector=vec, top_k=top_k, include_metadata=True)