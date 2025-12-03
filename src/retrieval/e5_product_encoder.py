from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import joblib

from src.config import RAW_DATA_DIR, E5_DIR, RANDOM_SEED

MODEL_NAME = "intfloat/e5-base"


def main():
    products = pd.read_csv(RAW_DATA_DIR / "products.csv")
    texts = products["description"].tolist()

    model = SentenceTransformer(MODEL_NAME)
    # E5 expects 'query: ...' or 'passage: ...'; use passage-style for items
    inputs = [f"passage: {t}" for t in texts]
    emb = model.encode(inputs, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    E5_DIR.mkdir(parents=True, exist_ok=True)
    np.save(E5_DIR / "product_e5_embeddings.npy", emb)
    products.to_csv(E5_DIR / "products_with_e5_index.csv", index=False)
    joblib.dump({"model_name": MODEL_NAME}, E5_DIR / "e5_meta.joblib")
    print(f"Saved E5 product embeddings to {E5_DIR}")


if __name__ == "__main__":
    main()

