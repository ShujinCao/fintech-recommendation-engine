# src/serving/inference.py

from typing import List

import pandas as pd

from src.retrieval.twotower_faiss import TwoTowerANN
from src.config import PROCESSED_DATA_DIR
import joblib
import lightgbm as lgb


class OnlineRecommender:
    def __init__(self):
        # ANN over multimodal two-tower embeddings
        print(">>> STEP 1: self.twotower_ann = TwoTowerANN()")
        self.twotower_ann = TwoTowerANN()
        print(">>> STEP 2: self.feature_pipeline, self.ranker")
        # ranker
        self.feature_pipeline = joblib.load(PROCESSED_DATA_DIR / "feature_pipeline.joblib")
        from src.config import RANKER_DIR

        self.ranker = lgb.Booster(model_file=str(RANKER_DIR / "ranker.txt"))

        print(">>> STEP 3: users, products")

        self.users = self.twotower_ann.users
        self.products = self.twotower_ann.products

    def _build_online_features(self, user_id: int, cand_df: pd.DataFrame) -> pd.DataFrame:
        # same logic as before: build tabular features for LightGBM
        user_row = self.users[self.users["user_id"] == user_id].iloc[0]
        user_repeat = pd.concat([user_row.to_frame().T] * len(cand_df), ignore_index=True)

        numeric_features = ["age", "income", "risk_tolerance", "credit_score", "apr",
                            "annual_fee", "min_income_required"]
        categorical_features = ["product_type", "has_mortgage", "has_auto_loan", "region"]

        merged = pd.DataFrame()
        merged[numeric_features] = pd.concat(
            [
                user_repeat[["age", "income", "risk_tolerance", "credit_score"]].reset_index(drop=True),
                cand_df[["apr", "annual_fee", "min_income_required"]].reset_index(drop=True),
            ],
            axis=1,
        )
        merged[categorical_features] = pd.concat(
            [
                user_repeat[["has_mortgage", "has_auto_loan", "region"]].reset_index(drop=True),
                cand_df[["product_type"]].reset_index(drop=True),
            ],
            axis=1,
        )
        return merged

    def recommend(self, user_id: int, k: int = 10) -> List[dict]:
        cand_df = self.twotower_ann.ann_candidates(user_id=user_id, top_k=100)

        feat_df = self._build_online_features(user_id, cand_df)
        X = self.feature_pipeline.transform(feat_df)
        scores = self.ranker.predict(X)
        cand_df = cand_df.copy()
        cand_df["rank_score"] = scores

        topk = cand_df.sort_values("rank_score", ascending=False).head(k)

        return topk[["product_id", "product_type", "apr", "risk_level", "annual_fee", "rank_score"]].to_dict(
            orient="records"
        )

