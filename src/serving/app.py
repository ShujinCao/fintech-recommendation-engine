from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.serving.inference import OnlineRecommender

app = FastAPI(title="FinTech Product Recommendation API")

recommender = OnlineRecommender()


class Recommendation(BaseModel):
    product_id: int
    product_type: str
    apr: float
    risk_level: int
    annual_fee: float
    rank_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=List[Recommendation])
def recommend(user_id: int, k: int = 10):
    try:
        return recommender.recommend(user_id=user_id, k=k)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"user_id={user_id} not found")

