# src/models/two_tower.py

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class MultiModalTwoTower(nn.Module):
    """
    Multimodal two-tower model:
      - User tower:
          - user_id embedding
          - sequence of recent product_ids (behavior) -> GRU encoder
      - Product tower:
          - product_id embedding
          - numeric metadata features (APR, fee, etc.)
          - E5 text embedding of product description

    Everything is learned jointly with a contrastive InfoNCE loss.
    """

    def __init__(
        self,
        num_users: int,
        num_products: int,
        prod_meta_dim: int,
        prod_e5_dim: int,
        user_hist_max_len: int,
        id_emb_dim: int = 32,
        hist_emb_dim: int = 32,
        meta_emb_dim: int = 32,
        e5_proj_dim: int = 64,
        tower_hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_products = num_products
        self.user_hist_max_len = user_hist_max_len

        # --- USER TOWER ---

        # user id embedding
        self.user_id_emb = nn.Embedding(num_users, id_emb_dim)

        # product id embedding for history sequence
        self.hist_prod_emb = nn.Embedding(num_products, hist_emb_dim)

        # GRU over user history product embeddings
        self.hist_gru = nn.GRU(
            input_size=hist_emb_dim,
            hidden_size=hist_emb_dim,
            batch_first=True,
        )

        # user tower MLP on [user_id_emb || hist_vec]
        user_input_dim = id_emb_dim + hist_emb_dim
        self.user_mlp = nn.Sequential(
            nn.Linear(user_input_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Linear(tower_hidden_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Linear(tower_hidden_dim, tower_hidden_dim),
        )

        # --- PRODUCT TOWER ---

        # product id embedding
        self.prod_id_emb = nn.Embedding(num_products, id_emb_dim)

        # numeric metadata projection
        self.prod_meta_proj = nn.Sequential(
            nn.Linear(prod_meta_dim, meta_emb_dim),
            nn.ReLU(),
        )

        # E5 text embedding projection
        self.prod_e5_proj = nn.Sequential(
            nn.Linear(prod_e5_dim, e5_proj_dim),
            nn.ReLU(),
        )

        # product tower MLP on [prod_id_emb || meta_emb || e5_emb]
        prod_input_dim = id_emb_dim + meta_emb_dim + e5_proj_dim
        self.prod_mlp = nn.Sequential(
            nn.Linear(prod_input_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Linear(tower_hidden_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Linear(tower_hidden_dim, tower_hidden_dim),
        )

    # ---------- USER TOWER FORWARD ----------

    def encode_user(
        self,
        user_ids: torch.Tensor,           # [B]
        hist_prod_ids: torch.Tensor,      # [B, L] padded with 0 when no item
        hist_lengths: Optional[torch.Tensor] = None,  # [B], optional true lengths
    ) -> torch.Tensor:
        """
        user_ids:       LongTensor [B]
        hist_prod_ids:  LongTensor [B, L]  (0 can be 'pad' product_id)
        hist_lengths:   LongTensor [B] or None
        """
        # id embedding
        u_id = self.user_id_emb(user_ids)  # [B, id_emb_dim]

        # history embeddings
        hist_emb = self.hist_prod_emb(hist_prod_ids)  # [B, L, hist_emb_dim]

        if hist_lengths is not None:
            # pack padded sequence if you want exact lengths
            packed = nn.utils.rnn.pack_padded_sequence(
                hist_emb,
                lengths=hist_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.hist_gru(packed)  # h_n: [1, B, hist_emb_dim]
            hist_vec = h_n.squeeze(0)       # [B, hist_emb_dim]
        else:
            # simple: just pass through GRU assuming full length
            _, h_n = self.hist_gru(hist_emb)
            hist_vec = h_n.squeeze(0)       # [B, hist_emb_dim]

        user_in = torch.cat([u_id, hist_vec], dim=-1)  # [B, id_emb_dim + hist_emb_dim]
        user_vec = self.user_mlp(user_in)              # [B, H]
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        return user_vec

    # ---------- PRODUCT TOWER FORWARD ----------

    def encode_product(
        self,
        product_ids: torch.Tensor,        # [B]
        prod_meta: torch.Tensor,          # [B, prod_meta_dim]
        prod_e5: torch.Tensor,            # [B, prod_e5_dim]
    ) -> torch.Tensor:
        p_id = self.prod_id_emb(product_ids)          # [B, id_emb_dim]
        p_meta = self.prod_meta_proj(prod_meta)       # [B, meta_emb_dim]
        p_e5 = self.prod_e5_proj(prod_e5)             # [B, e5_proj_dim]

        prod_in = torch.cat([p_id, p_meta, p_e5], dim=-1)  # [B, total_dim]
        prod_vec = self.prod_mlp(prod_in)                  # [B, H]
        prod_vec = F.normalize(prod_vec, p=2, dim=-1)
        return prod_vec

    # ---------- CONTRASTIVE LOSS (InfoNCE) ----------

    def info_nce_loss(
        self,
        user_ids: torch.Tensor,          # [B]
        hist_prod_ids: torch.Tensor,     # [B, L]
        hist_lengths: Optional[torch.Tensor],
        product_ids: torch.Tensor,       # [B]
        prod_meta: torch.Tensor,         # [B, prod_meta_dim]
        prod_e5: torch.Tensor,           # [B, prod_e5_dim]
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Within-batch InfoNCE: each (user_i, product_i) is positive,
        all (user_i, product_j != i) are negatives.
        """
        u = self.encode_user(user_ids, hist_prod_ids, hist_lengths)      # [B, H]
        p = self.encode_product(product_ids, prod_meta, prod_e5)         # [B, H]

        # similarity matrix [B, B]
        logits = (u @ p.T) / temperature
        labels = torch.arange(u.size(0), device=u.device)
        loss = F.cross_entropy(logits, labels)
        return loss

