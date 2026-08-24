"""
src/uplift_model.py

First uplift-modeling implementation for the Dynamic Pricing Agent project.

APPROACH: multi-treatment T-learner
------------------------------------
data_generation.py assigns discount_percentage via assign_discount(), which
samples independently of every customer/session feature (a fixed categorical
distribution, unconditional on X). That means the observed data behaves like
a randomized experiment: within each treatment arm, the factual conversion
rate conditional on X is an unbiased estimate of the true potential-outcome
conversion rate under that treatment. A T-learner -- one supervised binary
classifier per treatment arm, trained only on that arm's rows -- is the
natural, simplest-possible baseline under exactly this condition. It also
keeps the four counterfactual estimates fully interpretable (one model you
can inspect per discount level) before reaching for more complex estimators
(X-learner, causal forests, etc.) later in the project.

This script:
  1. Loads the observable transactions table (never touches ground_truth.csv
     for anything except final evaluation).
  2. Splits train/test at the CUSTOMER level (not the transaction level) so
     no customer's history leaks across the split.
  3. Fits one GradientBoostingClassifier pipeline per discount level on the
     training rows for that level only.
  4. Predicts all four potential conversion probabilities for every test
     transaction (regardless of which discount that transaction actually
     received).
  5. Converts predicted probabilities into predicted expected profit per
     discount, and picks the predicted profit-maximizing discount.
  6. Evaluates (a) each arm's probabilistic calibration/discrimination on
     its own factual test rows, (b) potential-outcome and uplift accuracy
     against the HELD-OUT ground-truth counterfactual probabilities
     (evaluation-only), and (c) economic regret versus the true
     profit-maximizing policy.

Ground-truth columns (segment, price_sensitivity, probability_*_discount,
true_incremental_lift_*) are loaded ONLY for step 6 evaluation, merged in
after predictions already exist, and are never part of X_train/X_test.

Does NOT train a 4-class classifier. Does NOT build the policy engine.
Does NOT do hyperparameter search.

Run from the project root:
    python src/uplift_model.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RAW_DIR = "data/raw"
TX_PATH = os.path.join(RAW_DIR, "synthetic_transactions.csv")
GT_PATH = os.path.join(RAW_DIR, "ground_truth.csv")

MODEL_DIR = "models/uplift"
OUT_DIR = "data/processed/uplift"

DISCOUNT_LEVELS = [0, 5, 10, 15]
TEST_SIZE = 0.20

# Observable-at-checkout features only. discount_percentage, transaction_id,
# customer_id, timestamp, and every ground-truth/evaluation-only column are
# excluded on purpose -- see module docstring.
CATEGORICAL_FEATURES = ["customer_type", "category", "device_type"]
NUMERICAL_FEATURES = [
    "previous_purchases", "previous_abandons", "days_since_last_purchase",
    "customer_lifetime_value", "historical_discount_rate",
    "cart_value", "items_count", "margin_percentage",
    "time_on_checkout_seconds", "pages_viewed", "payment_attempts",
    "hour", "day_of_week",
]
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COL = "converted"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def save_table(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=(df.index.name is not None))
    print(f"  [saved table] {path}")


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved plot] {path}")


# --------------------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------------------
def load_data():
    """Load the observable transactions table. ground_truth.csv is loaded
    separately, later, and only ever used for evaluation."""
    if not os.path.exists(TX_PATH):
        raise FileNotFoundError(
            f"Expected {TX_PATH}. Run this script from the project root "
            f"(the directory containing 'data/' and 'src/')."
        )
    tx = pd.read_csv(TX_PATH, parse_dates=["timestamp"])
    return tx


# --------------------------------------------------------------------------------------
# 2. CUSTOMER-LEVEL TRAIN/TEST SPLIT
# --------------------------------------------------------------------------------------
def split_by_customer(tx, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split on unique customer_id BEFORE any preprocessing/encoding/fitting,
    so no customer's sessions appear in both train and test."""
    customers = tx["customer_id"].unique()
    train_customers, test_customers = train_test_split(
        customers, test_size=test_size, random_state=random_state
    )
    train_df = tx[tx["customer_id"].isin(train_customers)].reset_index(drop=True)
    test_df = tx[tx["customer_id"].isin(test_customers)].reset_index(drop=True)

    overlap = set(train_df["customer_id"]) & set(test_df["customer_id"])
    assert len(overlap) == 0, "Customer leakage between train and test!"

    print(f"Train customers: {len(train_customers):,}  |  Train rows: {len(train_df):,}")
    print(f"Test customers:  {len(test_customers):,}  |  Test rows:  {len(test_df):,}")
    print(f"Customer overlap between splits: {len(overlap)} (must be 0)")

    return train_df, test_df


# --------------------------------------------------------------------------------------
# 3. PREPROCESSING (fit independently per treatment model)
# --------------------------------------------------------------------------------------
def build_preprocessor():
    """Fresh ColumnTransformer per call. One-hot encode categoricals with
    unknown-category safety; numeric features pass through untouched
    (tree-based model, so no scaling needed) after simple median imputation
    as a defensive measure."""
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer([
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ("num", numeric_pipe, NUMERICAL_FEATURES),
    ])


# --------------------------------------------------------------------------------------
# 4. TRAIN ONE MODEL PER TREATMENT ARM
# --------------------------------------------------------------------------------------
def train_treatment_model(train_df, discount_level, random_state=RANDOM_STATE):
    """Filter to this treatment's training rows, fit a fresh
    preprocessing + GradientBoostingClassifier pipeline. discount_percentage
    itself is never a feature -- it's constant within this subset, since the
    arm defines the treatment being modeled."""
    arm_df = train_df[train_df["discount_percentage"] == discount_level]
    X_train = arm_df[FEATURE_COLS]
    y_train = arm_df[TARGET_COL]

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=random_state,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, len(arm_df)


def train_all_models(train_df):
    models = {}
    n_train_per_arm = {}
    for d in DISCOUNT_LEVELS:
        print(f"Training model for discount = {d}% ...")
        pipeline, n_train = train_treatment_model(train_df, d)
        models[d] = pipeline
        n_train_per_arm[d] = n_train
        print(f"  trained on {n_train:,} rows")
    return models, n_train_per_arm


# --------------------------------------------------------------------------------------
# 5. PREDICT ALL FOUR POTENTIAL OUTCOMES FOR EVERY TEST TRANSACTION
# --------------------------------------------------------------------------------------
def predict_potential_outcomes(models, df):
    """For every row, predict P(convert) under EACH discount level using
    that level's model, regardless of which discount the row actually
    received. This is the whole point of the T-learner: full potential-
    outcome vectors from observable features alone."""
    X = df[FEATURE_COLS]
    preds = df.copy()
    for d in DISCOUNT_LEVELS:
        preds[f"pred_p{d}"] = models[d].predict_proba(X)[:, 1]
    return preds


# --------------------------------------------------------------------------------------
# 6. UPLIFT
# --------------------------------------------------------------------------------------
def calculate_uplift(df):
    df = df.copy()
    df["pred_uplift_5"] = df["pred_p5"] - df["pred_p0"]
    df["pred_uplift_10"] = df["pred_p10"] - df["pred_p0"]
    df["pred_uplift_15"] = df["pred_p15"] - df["pred_p0"]
    # incremental uplift between adjacent treatments
    df["pred_incremental_0_to_5"] = df["pred_uplift_5"]
    df["pred_incremental_5_to_10"] = df["pred_p10"] - df["pred_p5"]
    df["pred_incremental_10_to_15"] = df["pred_p15"] - df["pred_p10"]
    return df


# --------------------------------------------------------------------------------------
# 7. ECONOMIC OPTIMIZATION (same definition as the EDA)
# --------------------------------------------------------------------------------------
def calculate_expected_profit(df, prob_prefix="pred_p", profit_prefix="pred_expected_profit"):
    """expected_profit(d) = P(convert|d) * profit_if_converted(d), where
    cost = V*(1-m), price_after_discount = V*(1-d), profit_if_converted =
    price_after_discount - cost. margin_percentage is gross margin BEFORE
    discount, per the EDA's established convention."""
    df = df.copy()
    V = df["cart_value"]
    m = df["margin_percentage"] / 100.0
    cost = V * (1 - m)

    profit_cols = []
    for d in DISCOUNT_LEVELS:
        price_after_discount = V * (1 - d / 100.0)
        profit_if_converted = price_after_discount - cost
        p_convert = df[f"{prob_prefix}{d}"]
        col = f"{profit_prefix}_{d}"
        df[col] = p_convert * profit_if_converted
        profit_cols.append(col)

    df[f"{profit_prefix.replace('expected_profit', 'max_expected_profit')}"] = df[profit_cols].max(axis=1)
    optimal_col = profit_prefix.replace("expected_profit", "optimal_discount")
    df[optimal_col] = df[profit_cols].idxmax(axis=1).apply(
        lambda c: int(c.rsplit("_", 1)[-1])
    )
    return df


# --------------------------------------------------------------------------------------
# 8. EVALUATE PROBABILISTIC MODELS (per-arm, factual rows only)
# --------------------------------------------------------------------------------------
def evaluate_models(preds_df, n_train_per_arm):
    rows = []
    for d in DISCOUNT_LEVELS:
        arm = preds_df[preds_df["discount_percentage"] == d]
        y_true = arm[TARGET_COL].values
        y_pred = arm[f"pred_p{d}"].values

        n_test = len(arm)
        conv_rate = y_true.mean() if n_test > 0 else np.nan

        if n_test > 0 and len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_pred)
        else:
            auc = np.nan

        if n_test > 0:
            ll = log_loss(y_true, np.clip(y_pred, 1e-6, 1 - 1e-6), labels=[0, 1])
            brier = brier_score_loss(y_true, y_pred)
        else:
            ll, brier = np.nan, np.nan

        rows.append({
            "discount_level": d,
            "n_train": n_train_per_arm[d],
            "n_test": n_test,
            "conversion_rate": round(conv_rate, 4) if n_test > 0 else np.nan,
            "roc_auc": round(auc, 4) if not np.isnan(auc) else np.nan,
            "log_loss": round(ll, 4) if n_test > 0 else np.nan,
            "brier_score": round(brier, 4) if n_test > 0 else np.nan,
        })

    result = pd.DataFrame(rows).set_index("discount_level")
    print(result)
    save_table(result, "probabilistic_metrics.csv")
    return result


# --------------------------------------------------------------------------------------
# 9. EVALUATE UPLIFT QUALITY AGAINST HIDDEN GROUND TRUTH (evaluation only)
# --------------------------------------------------------------------------------------
def evaluate_uplift(preds_df, gt_df):
    """Merge ground-truth counterfactual probabilities onto the test
    predictions PURELY for scoring. This never influenced model training."""
    gt_cols = ["transaction_id", "segment", "price_sensitivity",
               "probability_no_discount", "probability_5_discount",
               "probability_10_discount", "probability_15_discount"]
    merged = preds_df.merge(gt_df[gt_cols], on="transaction_id", how="left", validate="one_to_one")

    merged = merged.rename(columns={
        "probability_no_discount": "true_p0",
        "probability_5_discount": "true_p5",
        "probability_10_discount": "true_p10",
        "probability_15_discount": "true_p15",
    })
    merged["true_lift_5"] = merged["true_p5"] - merged["true_p0"]
    merged["true_lift_10"] = merged["true_p10"] - merged["true_p0"]
    merged["true_lift_15"] = merged["true_p15"] - merged["true_p0"]

    def mae_rmse(a, b):
        err = a - b
        return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))

    rows = []
    for d in DISCOUNT_LEVELS:
        mae, rmse = mae_rmse(merged[f"pred_p{d}"], merged[f"true_p{d}"])
        rows.append({"quantity": f"potential_outcome_p{d}", "MAE": round(mae, 4), "RMSE": round(rmse, 4)})

    for d in [5, 10, 15]:
        mae, rmse = mae_rmse(merged[f"pred_uplift_{d}"], merged[f"true_lift_{d}"])
        rows.append({"quantity": f"uplift_{d}", "MAE": round(mae, 4), "RMSE": round(rmse, 4)})

    result = pd.DataFrame(rows).set_index("quantity")
    print(result)
    save_table(result, "uplift_error_metrics.csv")
    return merged, result


# --------------------------------------------------------------------------------------
# 10. EVALUATE ECONOMIC POLICY: PREDICTED vs TRUE OPTIMAL DISCOUNT
# --------------------------------------------------------------------------------------
def evaluate_economic_policy(merged):
    """Compute the TRUE profit-maximizing discount (from ground-truth
    probabilities) and compare against the model's predicted optimal
    discount. Report exact agreement rate and economic regret -- the more
    meaningful metric given the 15%-optimal class is extremely rare."""
    merged = calculate_expected_profit(merged, prob_prefix="true_p", profit_prefix="true_expected_profit")

    merged["agrees_with_true_optimal"] = (
        merged["pred_optimal_discount"] == merged["true_optimal_discount"]
    )
    agreement_rate = merged["agrees_with_true_optimal"].mean()

    def true_profit_at(row, d):
        return row[f"true_expected_profit_{d}"]

    merged["true_expected_profit_of_pred_discount"] = merged.apply(
        lambda r: true_profit_at(r, r["pred_optimal_discount"]), axis=1
    )
    merged["regret"] = merged["true_max_expected_profit"] - merged["true_expected_profit_of_pred_discount"]

    avg_regret = merged["regret"].mean()
    median_regret = merged["regret"].median()

    denom_safe = merged["true_max_expected_profit"].replace(0, np.nan).abs()
    regret_pct = (merged["regret"] / denom_safe) * 100
    mean_regret_pct = regret_pct[np.isfinite(regret_pct)].mean()

    summary = pd.DataFrame([{
        "exact_optimal_discount_agreement_rate": round(agreement_rate, 4),
        "average_regret": round(avg_regret, 4),
        "median_regret": round(median_regret, 4),
        "average_regret_pct_of_true_max_profit": round(mean_regret_pct, 2),
        "n_test_transactions": len(merged),
    }])
    print(summary.T)
    save_table(summary.set_index("n_test_transactions"), "economic_policy_evaluation.csv")

    print("\nAgreement rate by true optimal discount:")
    by_true = merged.groupby("true_optimal_discount")["agrees_with_true_optimal"].agg(["mean", "count"])
    print(by_true)
    save_table(by_true, "economic_policy_agreement_by_true_discount.csv")

    return merged, summary


# --------------------------------------------------------------------------------------
# 11. CALIBRATION
# --------------------------------------------------------------------------------------
def calibration_table(preds_df, n_bins=10):
    """Reliability table per treatment arm on that arm's factual test rows:
    mean predicted probability vs actual observed conversion rate, by bin
    of predicted probability."""
    tables = []
    for d in DISCOUNT_LEVELS:
        arm = preds_df[preds_df["discount_percentage"] == d].copy()
        if len(arm) < n_bins:
            continue
        arm["bin"] = pd.qcut(arm[f"pred_p{d}"], n_bins, duplicates="drop")
        g = arm.groupby("bin", observed=True).agg(
            mean_predicted=(f"pred_p{d}", "mean"),
            observed_rate=(TARGET_COL, "mean"),
            n=(TARGET_COL, "size"),
        ).reset_index(drop=True)
        g["discount_level"] = d
        tables.append(g)
    result = pd.concat(tables, ignore_index=True)
    save_table(result, "calibration_table.csv")
    return result


# --------------------------------------------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------------------------------------------
def make_visualizations(preds_df, econ_df, calib_df):
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Predicted vs true conversion probability per treatment
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, d in zip(axes.flat, DISCOUNT_LEVELS):
        ax.scatter(econ_df[f"true_p{d}"], econ_df[f"pred_p{d}"], alpha=0.15, s=8, color="#3182ce")
        lims = [0, 1]
        ax.plot(lims, lims, color="#c53030", linestyle="--", linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f"Discount {d}%")
        ax.set_xlabel("True P(convert)")
        ax.set_ylabel("Predicted P(convert)")
    fig.suptitle("Predicted vs True Conversion Probability")
    fig.tight_layout()
    save_fig(fig, "01_pred_vs_true_conversion_prob.png")

    # 2. Predicted vs true uplift for 5/10/15
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, d in zip(axes, [5, 10, 15]):
        ax.scatter(econ_df[f"true_lift_{d}"], econ_df[f"pred_uplift_{d}"], alpha=0.15, s=8, color="#805ad5")
        lims = [min(econ_df[f"true_lift_{d}"].min(), econ_df[f"pred_uplift_{d}"].min()),
                max(econ_df[f"true_lift_{d}"].max(), econ_df[f"pred_uplift_{d}"].max())]
        ax.plot(lims, lims, color="#c53030", linestyle="--", linewidth=1)
        ax.set_title(f"Uplift {d}%")
        ax.set_xlabel("True uplift")
        ax.set_ylabel("Predicted uplift")
    fig.suptitle("Predicted vs True Uplift")
    fig.tight_layout()
    save_fig(fig, "02_pred_vs_true_uplift.png")

    # 3. Distribution of predicted optimal discounts
    fig, ax = plt.subplots(figsize=(6, 4))
    econ_df["pred_optimal_discount"].value_counts(normalize=True).reindex(DISCOUNT_LEVELS, fill_value=0)\
        .mul(100).plot(kind="bar", ax=ax, color="#2f855a")
    ax.set_xticklabels([f"{d}%" for d in DISCOUNT_LEVELS], rotation=0)
    ax.set_title("Predicted Optimal Discount Distribution")
    ax.set_ylabel("% of test transactions")
    save_fig(fig, "03_pred_optimal_discount_distribution.png")

    # 4. Distribution of true optimal discounts
    fig, ax = plt.subplots(figsize=(6, 4))
    econ_df["true_optimal_discount"].value_counts(normalize=True).reindex(DISCOUNT_LEVELS, fill_value=0)\
        .mul(100).plot(kind="bar", ax=ax, color="#b7791f")
    ax.set_xticklabels([f"{d}%" for d in DISCOUNT_LEVELS], rotation=0)
    ax.set_title("True Optimal Discount Distribution")
    ax.set_ylabel("% of test transactions")
    save_fig(fig, "04_true_optimal_discount_distribution.png")

    # 5. Predicted vs true expected profit, mean by discount level
    fig, ax = plt.subplots(figsize=(7, 4))
    pred_means = [econ_df[f"pred_expected_profit_{d}"].mean() for d in DISCOUNT_LEVELS]
    true_means = [econ_df[f"true_expected_profit_{d}"].mean() for d in DISCOUNT_LEVELS]
    x = np.arange(len(DISCOUNT_LEVELS))
    width = 0.35
    ax.bar(x - width / 2, pred_means, width, label="Predicted", color="#3182ce")
    ax.bar(x + width / 2, true_means, width, label="True", color="#c53030")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}%" for d in DISCOUNT_LEVELS])
    ax.set_ylabel("Mean expected profit")
    ax.set_title("Predicted vs True Expected Profit by Discount")
    ax.legend()
    save_fig(fig, "05_pred_vs_true_expected_profit.png")

    # 6. Calibration plot per treatment
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, d in zip(axes.flat, DISCOUNT_LEVELS):
        sub = calib_df[calib_df["discount_level"] == d]
        if len(sub) == 0:
            continue
        ax.plot(sub["mean_predicted"], sub["observed_rate"], marker="o", color="#2f855a")
        lims = [0, 1]
        ax.plot(lims, lims, color="#c53030", linestyle="--", linewidth=1)
        ax.set_xlim(0, max(0.05, sub["mean_predicted"].max() * 1.1))
        ax.set_ylim(0, max(0.05, sub["observed_rate"].max() * 1.1))
        ax.set_title(f"Discount {d}%")
        ax.set_xlabel("Mean predicted P(convert)")
        ax.set_ylabel("Observed conversion rate")
    fig.suptitle("Calibration (reliability) by Treatment Arm")
    fig.tight_layout()
    save_fig(fig, "06_calibration_by_treatment.png")


# --------------------------------------------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------------------------------------------
def save_outputs(models, econ_df):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    for d in DISCOUNT_LEVELS:
        path = os.path.join(MODEL_DIR, f"model_discount_{d}.joblib")
        joblib.dump(models[d], path)
        print(f"  [saved model] {path}")

    meta = {
        "feature_cols": FEATURE_COLS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "discount_levels": DISCOUNT_LEVELS,
        "target_col": TARGET_COL,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    meta_path = os.path.join(MODEL_DIR, "feature_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [saved metadata] {meta_path}")

    # Build the final test_predictions.csv per the spec: observable features +
    # predictions + clearly-labeled ground-truth columns (evaluation only,
    # never fed back into training).
    observable_cols = ["transaction_id", "customer_id", "discount_percentage", TARGET_COL] + FEATURE_COLS
    pred_cols = (
        [f"pred_p{d}" for d in DISCOUNT_LEVELS]
        + ["pred_uplift_5", "pred_uplift_10", "pred_uplift_15"]
        + ["pred_incremental_0_to_5", "pred_incremental_5_to_10", "pred_incremental_10_to_15"]
        + [f"pred_expected_profit_{d}" for d in DISCOUNT_LEVELS]
        + ["pred_optimal_discount", "pred_max_expected_profit"]
    )
    gt_eval_cols = (
        ["segment", "price_sensitivity"]
        + [f"true_p{d}" for d in DISCOUNT_LEVELS]
        + ["true_lift_5", "true_lift_10", "true_lift_15"]
        + [f"true_expected_profit_{d}" for d in DISCOUNT_LEVELS]
        + ["true_optimal_discount", "true_max_expected_profit", "regret", "agrees_with_true_optimal"]
    )
    rename_gt = {c: f"gt_{c}" for c in gt_eval_cols}

    final_cols = observable_cols + pred_cols + gt_eval_cols
    final_cols = [c for c in final_cols if c in econ_df.columns]
    out_df = econ_df[final_cols].rename(columns=rename_gt)

    out_path = os.path.join(OUT_DIR, "test_predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  [saved predictions] {out_path}  ({len(out_df):,} rows)")


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    hr("LOAD DATA")
    tx = load_data()
    print(f"Loaded {len(tx):,} transactions.")

    hr("CUSTOMER-LEVEL TRAIN/TEST SPLIT")
    train_df, test_df = split_by_customer(tx)

    hr("TRAIN T-LEARNER (one GradientBoostingClassifier per discount level)")
    models, n_train_per_arm = train_all_models(train_df)

    hr("PREDICT POTENTIAL OUTCOMES ON TEST SET")
    preds_df = predict_potential_outcomes(models, test_df)
    preds_df = calculate_uplift(preds_df)
    preds_df = calculate_expected_profit(preds_df, prob_prefix="pred_p", profit_prefix="pred_expected_profit")
    print(preds_df[["transaction_id", "pred_p0", "pred_p5", "pred_p10", "pred_p15",
                     "pred_optimal_discount", "pred_max_expected_profit"]].head())

    hr("EVALUATE PROBABILISTIC MODELS (per arm, factual test rows)")
    prob_metrics = evaluate_models(preds_df, n_train_per_arm)

    hr("EVALUATE UPLIFT QUALITY AGAINST HIDDEN GROUND TRUTH")
    gt = pd.read_csv(GT_PATH)
    merged, uplift_metrics = evaluate_uplift(preds_df, gt)

    hr("EVALUATE ECONOMIC POLICY (regret vs true profit-optimal discount)")
    merged, econ_summary = evaluate_economic_policy(merged)

    hr("CALIBRATION")
    calib_df = calibration_table(preds_df)
    print(calib_df)

    hr("VISUALIZATIONS")
    make_visualizations(preds_df, merged, calib_df)

    hr("SAVE OUTPUTS")
    save_outputs(models, merged)

    hr("DONE")
    print(f"Models saved under: {MODEL_DIR}/")
    print(f"Evaluation tables, plots, and test_predictions.csv saved under: {OUT_DIR}/")


if __name__ == "__main__":
    main()