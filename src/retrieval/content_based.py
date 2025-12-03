import faiss
import numpy as np
import pandas as pd

from src.config import E5_DIR


class E5Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(E5_DIR / "product_e5.faissindex"))
        self.emb = np.load(E5_DIR / "product_e5_embeddings.npy")
        self.products = pd.read_csv(E5_DIR / "products_e5.csv")

    def search(self, query_embedding: np.ndarray, top_k: int = 20):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1).astype("float32")

        scores, idx = self.index.search(query_embedding, top_k)
        results = self.products.iloc[idx[0]].copy()
        results["score"] = scores[0]
        return results

