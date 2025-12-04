from fastapi import FastAPI

from src.serving.inference import OnlineRecommender

app = FastAPI(title="FinTech Product Recommendation API")

print(">>> LOADING RECOMMENDER...")
try:
    recommender = OnlineRecommender()
    print(">>> RECOMMENDER LOADED.")
except Exception as e:
    print(">>> FAILED TO LOAD RECOMMENDER <<<")
    print("ERROR:", e)
    recommender = None


@app.get("/health")
def health():
    if recommender is None:
        return {"status": "error", "detail": "Recommender failed to load"}
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if recommender is None:
        return {"error": "Recommender not available"}
    return recommender.recommend(user_id=user_id, k=k)

