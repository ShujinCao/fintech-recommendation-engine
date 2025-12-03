# src/models/train_two_tower_multimodal.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim

import joblib

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    E5_DIR,
    TWO_TOWER_DIR,
    RANDOM_SEED,
    EMBEDDING_DIM,
    BATCH_SIZE,
    N_EPOCHS,
    INFO_NCE_TEMPERATURE,
)
from src.models.two_tower import MultiModalTwoTower

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class MultiModalTwoTowerDataset(Dataset):
    """
    Dataset for multimodal two-tower:
      - uses interactions_for_two_tower.csv
      - needs:
          * user_id
          * product_id
          * product metadata
          * product E5 embedding
          * user behavior sequence (recent product_ids)
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        products_df: pd.DataFrame,
        prod_e5_emb: np.ndarray,
        user_hist_dict: dict[int, list[int]],
        user_hist_max_len: int,
    ):
        self.df = interactions_df[interactions_df["label"] == 1].reset_index(drop=True)
        self.products = products_df.set_index("product_id")
        self.prod_e5_emb = prod_e5_emb
        self.user_hist_dict = user_hist_dict
        self.user_hist_max_len = user_hist_max_len

        # we’ll define product metadata columns explicitly
        self.prod_meta_cols = ["apr", "risk_level", "annual_fee", "min_income_required"]

    def __len__(self):
        return len(self.df)

    def _get_hist(self, user_id: int):
        hist = self.user_hist_dict.get(user_id, [])
        # keep only last L
        hist = hist[-self.user_hist_max_len :]
        # pad on the left with 0 (assuming product 0 exists)
        pad_len = self.user_hist_max_len - len(hist)
        padded = [0] * pad_len + hist
        length = len(hist)
        return np.array(padded, dtype="int64"), length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = int(row["user_id"])
        product_id = int(row["product_id"])

        # user history
        hist_ids, hist_len = self._get_hist(user_id)

        # product metadata
        p_meta = self.products.loc[product_id, self.prod_meta_cols].values.astype("float32")

        # product e5 embedding
        p_e5 = self.prod_e5_emb[product_id].astype("float32")

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(hist_ids, dtype=torch.long),
            torch.tensor(hist_len, dtype=torch.long),
            torch.tensor(product_id, dtype=torch.long),
            torch.tensor(p_meta, dtype=torch.float32),
            torch.tensor(p_e5, dtype=torch.float32),
        )


def build_user_history(interactions: pd.DataFrame, user_hist_max_len: int) -> dict[int, list[int]]:
    """
    Very simple behavior history:
      - sort interactions by some notion of time (here index order)
      - accumulate past products per user
    """
    user_hist: dict[int, list[int]] = {}
    for _, row in interactions.sort_index().iterrows():
        u = int(row["user_id"])
        p = int(row["product_id"])
        if u not in user_hist:
            user_hist[u] = []
        # history is 'before' current product; we add current after
        user_hist[u].append(p)
        # we do not limit here; we trim later in dataset
    return user_hist


def main(
    user_hist_max_len: int = 20,
    tower_hidden_dim: int = 128,
):
    # load raw + processed
    interactions = pd.read_csv(PROCESSED_DATA_DIR / "interactions_for_two_tower.csv")
    products = pd.read_csv(RAW_DATA_DIR / "products.csv")
    prod_e5_emb = np.load(E5_DIR / "product_e5_embeddings.npy")  # [num_products, e5_dim]

    num_users = interactions["user_id"].max() + 1
    num_products = products["product_id"].max() + 1
    prod_meta_dim = 4                     # apr, risk_level, annual_fee, min_income_required
    prod_e5_dim = prod_e5_emb.shape[1]

    # build user behavior histories
    user_hist_dict = build_user_history(interactions, user_hist_max_len=user_hist_max_len)

    dataset = MultiModalTwoTowerDataset(
        interactions_df=interactions,
        products_df=products,
        prod_e5_emb=prod_e5_emb,
        user_hist_dict=user_hist_dict,
        user_hist_max_len=user_hist_max_len,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalTwoTower(
        num_users=num_users,
        num_products=num_products,
        prod_meta_dim=prod_meta_dim,
        prod_e5_dim=prod_e5_dim,
        user_hist_max_len=user_hist_max_len,
        tower_hidden_dim=tower_hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        n = 0
        for (
            user_ids,
            hist_ids,
            hist_lens,
            product_ids,
            prod_meta,
            prod_e5,
        ) in dataloader:
            user_ids = user_ids.to(device)
            hist_ids = hist_ids.to(device)
            hist_lens = hist_lens.to(device)
            product_ids = product_ids.to(device)
            prod_meta = prod_meta.to(device)
            prod_e5 = prod_e5.to(device)

            optimizer.zero_grad()
            loss = model.info_nce_loss(
                user_ids=user_ids,
                hist_prod_ids=hist_ids,
                hist_lengths=hist_lens,
                product_ids=product_ids,
                prod_meta=prod_meta,
                prod_e5=prod_e5,
                temperature=INFO_NCE_TEMPERATURE,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * user_ids.size(0)
            n += user_ids.size(0)

        avg_loss = total_loss / n
        print(f"Epoch {epoch+1}/{N_EPOCHS} - InfoNCE loss: {avg_loss:.4f}")

    TWO_TOWER_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), TWO_TOWER_DIR / "two_tower_multimodal.pt")
    # save config-like info
    joblib.dump(
        {
            "num_users": num_users,
            "num_products": num_products,
            "prod_meta_dim": prod_meta_dim,
            "prod_e5_dim": prod_e5_dim,
            "user_hist_max_len": user_hist_max_len,
        },
        TWO_TOWER_DIR / "two_tower_meta.joblib",
    )
    print("Saved multimodal two-tower model to", TWO_TOWER_DIR)

    # precompute product embeddings for ANN
    model.eval()
    with torch.no_grad():
        prod_meta = products[["apr", "risk_level", "annual_fee", "min_income_required"]].values.astype("float32")
        prod_meta_t = torch.from_numpy(prod_meta).to(device)
        prod_ids_t = torch.arange(num_products, dtype=torch.long, device=device)
        prod_e5_t = torch.from_numpy(prod_e5_emb).to(device)

        prod_vecs = []
        batch_size = 1024
        for i in range(0, num_products, batch_size):
            sl = slice(i, i + batch_size)
            v = model.encode_product(
                product_ids=prod_ids_t[sl],
                prod_meta=prod_meta_t[sl],
                prod_e5=prod_e5_t[sl],
            )
            prod_vecs.append(v.cpu().numpy())
        prod_vecs = np.vstack(prod_vecs)  # [num_products, H]

    np.save(TWO_TOWER_DIR / "product_embeddings_multimodal.npy", prod_vecs)
    print("Saved multimodal product embeddings to", TWO_TOWER_DIR)


if __name__ == "__main__":
    main()

