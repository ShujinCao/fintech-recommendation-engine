import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.config import PROCESSED_DATA_DIR, RANKER_DIR


def main():
    data = joblib.load(PROCESSED_DATA_DIR / "ranker_dataset.joblib")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": ["auc"],
        "learning_rate": 0.05,
        "num_leaves": 63,
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
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_val_pred)
    print(f"Validation AUC: {auc:.4f}")

    RANKER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(RANKER_DIR / "ranker.txt"))
    print("Saved LightGBM ranker to", RANKER_DIR)


if __name__ == "__main__":
    main()

