# src/retrieval/twotower_faiss.py

from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
import joblib
import torch

from src.config import RAW_DATA_DIR, TWO_TOWER_DIR
from src.models.two_tower import MultiModalTwoTower


class TwoTowerANN:
    """
    ANN retrieval over multimodal two-tower product embeddings.
    """

    def __init__(self):
        print(">>> TwoTowerANN STEP 1: Loading metadata")
        self.meta = joblib.load(TWO_TOWER_DIR / "two_tower_meta.joblib")
        self.num_users = self.meta["num_users"]
        self.num_products = self.meta["num_products"]
        self.prod_meta_dim = self.meta["prod_meta_dim"]
        self.prod_e5_dim = self.meta["prod_e5_dim"]
        self.user_hist_max_len = self.meta["user_hist_max_len"]

        print(">>> TwoTowerANN STEP 2: Loading products and users")
        self.products = pd.read_csv(RAW_DATA_DIR / "products.csv")
        self.users = pd.read_csv(RAW_DATA_DIR / "users.csv")
        self.prod_meta_cols = ["apr", "risk_level", "annual_fee", "min_income_required"]
        print(">>> TwoTowerANN STEP 3: Building model")
        # load multimodal two-tower model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiModalTwoTower(
            num_users=self.num_users,
            num_products=self.num_products,
            prod_meta_dim=self.prod_meta_dim,
            prod_e5_dim=self.prod_e5_dim,
            user_hist_max_len=self.user_hist_max_len,
            id_emb_dim=self.meta["id_emb_dim"],
            hist_emb_dim=self.meta["hist_emb_dim"],
            meta_emb_dim=self.meta["meta_emb_dim"],
            e5_proj_dim=self.meta["e5_proj_dim"],            
            tower_hidden_dim=self.meta["tower_hidden_dim"],
        ).to(self.device)

        print(">>> TwoTowerANN STEP 4: Loading model weights")
        print("again to make sure")
        self.model.load_state_dict(torch.load(TWO_TOWER_DIR / "two_tower_multimodal.pt", map_location=self.device))
        print(">>> TwoTowerANN STEP 4.1: eval")
        self.model.eval()

        print(">>> TwoTowerANN STEP 5: Loading product embeddings")
        # load product embeddings & build FAISS ANN index
        prod_vecs = np.load(TWO_TOWER_DIR / "product_embeddings_multimodal.npy").astype("float32")
        print(">>> TwoTowerANN STEP 6: Building FAISS index")

        self.index = faiss.IndexFlatIP(prod_vecs.shape[1])  # inner product over normalized vectors ~ cosine
        self.index.add(prod_vecs)
        print(">>> TwoTowerANN STEP 7: Building user histories")
        # for user history: build a simple map from user->past products
        inter = pd.read_csv(RAW_DATA_DIR / "interactions.csv")
        self.user_hist = (
            inter.groupby("user_id")["product_id"]
            .apply(list)
            .to_dict()
        )  


        # load E5 embeddings for products (used only to rebuild product tower if needed)
        print(">>> TwoTowerANN STEP 8: Loading E5 embeddings")
        self.prod_e5_emb = np.load(TWO_TOWER_DIR.parent / "e5" / "product_e5_embeddings.npy").astype("float32")
        print(">>> TwoTowerANN INITIALIZATION COMPLETE")

    def _get_user_hist(self, user_id: int):
        hist = self.user_hist.get(user_id, [])
        hist = hist[-self.user_hist_max_len :]
        pad_len = self.user_hist_max_len - len(hist)
        padded = [0] * pad_len + hist
        length = len(hist)
        return np.array(padded, dtype="int64"), length

    def user_vector(self, user_id: int) -> np.ndarray:
        if user_id >= self.num_users:
            raise ValueError(f"user_id {user_id} out of range")

        hist_ids, hist_len = self._get_user_hist(user_id)
        user_ids_t = torch.tensor([user_id], dtype=torch.long, device=self.device)
        hist_ids_t = torch.tensor(hist_ids.reshape(1, -1), dtype=torch.long, device=self.device)
        hist_len_t = torch.tensor([hist_len], dtype=torch.long, device=self.device)

        with torch.no_grad():
            u_vec = self.model.encode_user(
                user_ids=user_ids_t,
                hist_prod_ids=hist_ids_t,
                hist_lengths=hist_len_t,
            )
        return u_vec.cpu().numpy()[0].astype("float32")

    def ann_candidates(self, user_id: int, top_k: int = 50) -> pd.DataFrame:
        u_vec = self.user_vector(user_id)
        u_vec = u_vec.reshape(1, -1)
        scores, idx = self.index.search(u_vec, top_k)
        idx = idx[0]
        scores = scores[0]
        # map FAISS index -> product_id (they are aligned with row index 0..num_products-1)
        products = self.products.iloc[idx].copy()
        products["collab_score"] = scores
        return products

