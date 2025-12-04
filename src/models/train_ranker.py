import joblib
import lightgbm as lgb
import pandas as pd
from src.config import PROCESSED_DATA_DIR, RANKER_DIR


def main():
    # Load processed features
    data = joblib.load(PROCESSED_DATA_DIR / "ranker_dataset.joblib")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    # Load the interactions to compute group sizes
    # Each group = 1 user’s interaction set
    interactions = pd.read_csv(PROCESSED_DATA_DIR / "interactions_for_two_tower.csv")

    # Reconstruct group sizes
    train_inter = interactions.iloc[: len(y_train)]
    val_inter = interactions.iloc[len(y_train):]

    train_group = train_inter.groupby("user_id").size().tolist()
    val_group = val_inter.groupby("user_id").size().tolist()

    # Build LightGBM datasets
    train_ds = lgb.Dataset(X_train, label=y_train, group=train_group)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_group)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_ds,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "valid"],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50)]
    )

    # Save model
    RANKER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(RANKER_DIR / "ranker.txt"))

    print("LambdaRank model trained and saved.")


if __name__ == "__main__":
    main()

