import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, N_USERS, N_PRODUCTS, N_INTERACTIONS, RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


def generate_users(n_users: int) -> pd.DataFrame:
    user_ids = np.arange(n_users)
    ages = rng.integers(18, 75, size=n_users)
    incomes = rng.normal(70_000, 30_000, size=n_users).clip(20_000, 300_000)
    risk_tolerance = rng.integers(1, 6, size=n_users)  # 1-5
    credit_score = rng.normal(700, 50, size=n_users).clip(500, 850)
    has_mortgage = rng.integers(0, 2, size=n_users)
    has_auto_loan = rng.integers(0, 2, size=n_users)
    region = rng.choice(["west", "midwest", "south", "northeast"], size=n_users)

    return pd.DataFrame({
        "user_id": user_ids,
        "age": ages,
        "income": incomes.astype(int),
        "risk_tolerance": risk_tolerance,
        "credit_score": credit_score.astype(int),
        "has_mortgage": has_mortgage,
        "has_auto_loan": has_auto_loan,
        "region": region,
    })


def generate_products(n_products: int) -> pd.DataFrame:
    product_ids = np.arange(n_products)
    product_types = ["installment_loan", "savings_account", "credit_card", "insurance"]
    product_type = rng.choice(product_types, size=n_products, p=[0.3, 0.3, 0.25, 0.15])

    base_apr = {
        "installment_loan": 0.15,
        "savings_account": 0.02,
        "credit_card": 0.20,
        "insurance": 0.0,
    }
    risk_level_map = {"installment_loan": 3, "savings_account": 1, "credit_card": 4, "insurance": 2}

    apr = np.array([base_apr[t] for t in product_type]) + rng.normal(0, 0.01, size=n_products)
    risk_level = np.array([risk_level_map[t] for t in product_type])
    annual_fee = rng.choice([0, 25, 50, 100], size=n_products, p=[0.5, 0.3, 0.15, 0.05])
    min_income_required = rng.choice([20_000, 40_000, 60_000, 80_000], size=n_products)

    # synthetic descriptions for E5
    descriptions = []
    for t, apr_i, fee_i in zip(product_type, apr, annual_fee):
        if t == "installment_loan":
            desc = f"Installment loan with approximately {apr_i*100:.1f}% APR and annual fee {fee_i} USD, suitable for medium to large purchases."
        elif t == "savings_account":
            desc = f"Savings account offering about {apr_i*100:.2f}% interest rate, ideal for low-risk savings and emergency funds."
        elif t == "credit_card":
            desc = f"Credit card product with around {apr_i*100:.1f}% APR and annual fee {fee_i} USD, designed for everyday spending."
        else:
            desc = "Insurance policy product providing financial protection against unexpected events."
        descriptions.append(desc)

    return pd.DataFrame({
        "product_id": product_ids,
        "product_type": product_type,
        "apr": apr.clip(0, 0.35),
        "risk_level": risk_level,
        "annual_fee": annual_fee,
        "min_income_required": min_income_required,
        "description": descriptions,
    })


def generate_interactions(
    users: pd.DataFrame,
    products: pd.DataFrame,
    n_interactions: int
) -> pd.DataFrame:
    user_ids = users["user_id"].values
    product_ids = products["product_id"].values

    sampled_users = rng.choice(user_ids, size=n_interactions, replace=True)
    sampled_products = rng.choice(product_ids, size=n_interactions, replace=True)

    user_income = users.set_index("user_id").loc[sampled_users, "income"].values
    user_risk = users.set_index("user_id").loc[sampled_users, "risk_tolerance"].values
    product_risk = products.set_index("product_id").loc[sampled_products, "risk_level"].values

    # click propensity: income & risk alignment
    base_click = 0.1 + 0.2 * (user_income / user_income.max()) + 0.1 * (user_risk / 5) - 0.05 * np.abs(user_risk - product_risk)
    click_prob = np.clip(base_click, 0.02, 0.8)
    clicked = rng.binomial(1, click_prob)

    # application probability given click
    app_prob = np.clip(0.2 + 0.3 * (user_income / user_income.max()), 0.05, 0.9)
    applied = rng.binomial(1, clicked * app_prob)

    # approval probability
    user_credit = users.set_index("user_id").loc[sampled_users, "credit_score"].values
    approval_score = (user_credit - 650) / 200 - 0.1 * (product_risk - 2)
    approval_prob = np.clip(0.2 + 0.4 * approval_score, 0.01, 0.95)
    approved = rng.binomial(1, applied * approval_prob)

    return pd.DataFrame({
        "user_id": sampled_users,
        "product_id": sampled_products,
        "clicked": clicked,
        "applied": applied,
        "approved": approved,
    })


def main():
    users = generate_users(N_USERS)
    products = generate_products(N_PRODUCTS)
    interactions = generate_interactions(users, products, N_INTERACTIONS)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    users.to_csv(RAW_DATA_DIR / "users.csv", index=False)
    products.to_csv(RAW_DATA_DIR / "products.csv", index=False)
    interactions.to_csv(RAW_DATA_DIR / "interactions.csv", index=False)
    print(f"Saved synthetic data to {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()

