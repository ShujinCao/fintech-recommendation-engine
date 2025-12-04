import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED


def main():
    users = pd.read_csv(RAW_DATA_DIR / "users.csv")
    products = pd.read_csv(RAW_DATA_DIR / "products.csv")
    interactions = pd.read_csv(RAW_DATA_DIR / "interactions.csv")

    df = interactions.merge(users, on="user_id", how="left", suffixes=("", "_user"))
    df = df.merge(products, on="product_id", how="left", suffixes=("", "_product"))

    # label: you can choose clicked or approved; I'll pick 'approved' to reflect value
    df["label"] = df["approved"]

    numeric_features = ["age", "income", "risk_tolerance", "credit_score", "apr",
                        "annual_fee", "min_income_required"]
    categorical_features = ["product_type", "has_mortgage", "has_auto_loan", "region"]

    X = df[numeric_features + categorical_features]
    y = df["label"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train_trans = pipeline.fit_transform(X_train)
    X_val_trans = pipeline.transform(X_val)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PROCESSED_DATA_DIR / "feature_pipeline.joblib")
    joblib.dump(
        {
            "X_train": X_train_trans,
            "y_train": y_train.values,
            "X_val": X_val_trans,
            "y_val": y_val.values,
        },
        PROCESSED_DATA_DIR / "ranker_dataset.joblib",
    )

    # store subset for id-based two-tower training & evaluation
    df[["user_id", "product_id", "label"]].to_csv(
        PROCESSED_DATA_DIR / "interactions_for_two_tower.csv", index=False
    )

    print("Saved feature pipeline and ranker dataset to", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()

