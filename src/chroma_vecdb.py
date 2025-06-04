import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, name, persist_directory="chroma_db"):
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        self.client = chromadb.Client(settings)
        self.name = name
        self.collection = self.client.get_or_create_collection(name)

    def insert(self, data):
        ids = [str(item["id"]) for item in data]
        embeddings = [item["vector"] for item in data]
        metadatas = [{"name": item["name"]} for item in data]
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, data, top_k):
        results = []
        query = self.collection.query(data, n_results=top_k, include=["distances", "metadatas"])
        for distances, metadatas in zip(query["distances"], query["metadatas"]):
            hits = []
            for dist, meta in zip(distances, metadatas):
                hits.append({"entity": {"name": meta["name"]}, "distance": dist})
            results.append(hits)
        return results

    def get(self, ids):
        docs = self.collection.get(ids=[str(i) for i in ids], include=["embeddings", "metadatas"])
        results = []
        for id_, emb, meta in zip(docs["ids"], docs["embeddings"], docs["metadatas"]):
            results.append({"id": int(id_), "vector": emb, "name": meta["name"]})
        return results

    def count(self):
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.name)
        self.collection = self.client.get_or_create_collection(self.name)
