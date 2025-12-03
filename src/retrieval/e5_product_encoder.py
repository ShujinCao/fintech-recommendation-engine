from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import joblib
import faiss
from pathlib import Path

from src.config import RAW_DATA_DIR, E5_DIR

MODEL_NAME = "intfloat/e5-base"


def main():
    products = pd.read_csv(RAW_DATA_DIR / "products.csv")
    texts = [f"passage: {t}" for t in products["description"].tolist()]

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    # ----- FAISS INDEX -----
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product similarity
    index.add(emb)

    # Save embeddings + index + metadata
    E5_DIR.mkdir(parents=True, exist_ok=True)
    np.save(E5_DIR / "product_e5_embeddings.npy", emb)
    faiss.write_index(index, str(E5_DIR / "product_e5.faissindex"))
    products.to_csv(E5_DIR / "products_e5.csv", index=False)
    joblib.dump({"model_name": MODEL_NAME}, E5_DIR / "meta.joblib")

    print("E5 embeddings + FAISS index saved to", E5_DIR)


if __name__ == "__main__":
    main()

