"""
src/eda.py

Exploratory Data Analysis + synthetic-environment validation for the
Dynamic Pricing Agent project.

This script does NOT train any ML model. Its purpose is to check whether
data/raw/synthetic_transactions.csv + data/raw/ground_truth.csv together
define a dynamic-pricing problem that is (a) statistically sane and
(b) non-trivial enough to be worth solving with an uplift model.

IMPORTANT — ground_truth.csv columns (segment, price_sensitivity,
base_conversion_probability, probability_*_discount, true_incremental_lift_*)
are EVALUATION-ONLY / latent simulator outputs. They are used here purely to
judge whether the synthetic environment is well-posed. They are never
combined with synthetic_transactions.csv to form an ML feature set anywhere
in this script.

Run from the project root:
    python src/eda.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

RAW_DIR = "data/raw"
TX_PATH = os.path.join(RAW_DIR, "synthetic_transactions.csv")
GT_PATH = os.path.join(RAW_DIR, "ground_truth.csv")

OUT_DIR = "data/processed/eda"
DISCOUNT_LEVELS = [0, 5, 10, 15]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved plot] {path}")


def save_table(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path)
    print(f"  [saved table] {path}")


# --------------------------------------------------------------------------------------
# 0. LOAD
# --------------------------------------------------------------------------------------
def load_data():
    if not os.path.exists(TX_PATH) or not os.path.exists(GT_PATH):
        raise FileNotFoundError(
            f"Expected {TX_PATH} and {GT_PATH}. Run this script from the project root "
            f"(the directory containing 'data/' and 'src/')."
        )
    tx = pd.read_csv(TX_PATH, parse_dates=["timestamp"])
    gt = pd.read_csv(GT_PATH)
    return tx, gt


# --------------------------------------------------------------------------------------
# 1. BASIC DATA VALIDATION
# --------------------------------------------------------------------------------------
def basic_validation(tx, gt):
    hr("1. BASIC DATA VALIDATION")

    print(f"Transactions rows: {len(tx):,}")
    print(f"Ground truth rows: {len(gt):,}")
    print(f"Unique customers: {tx['customer_id'].nunique():,}")
    print(f"Date range: {tx['timestamp'].min()}  ->  {tx['timestamp'].max()}")

    print("\nMissing values (synthetic_transactions.csv):")
    miss_tx = tx.isnull().sum()
    print(miss_tx[miss_tx > 0] if miss_tx.sum() else "  None")

    print("\nMissing values (ground_truth.csv):")
    miss_gt = gt.isnull().sum()
    print(miss_gt[miss_gt > 0] if miss_gt.sum() else "  None")

    dup_tx = tx["transaction_id"].duplicated().sum()
    dup_gt = gt["transaction_id"].duplicated().sum()
    print(f"\nDuplicate transaction_id in transactions: {dup_tx}")
    print(f"Duplicate transaction_id in ground_truth: {dup_gt}")

    print("\nNumerical summary statistics (synthetic_transactions.csv):")
    num_summary = tx.select_dtypes(include=[np.number]).describe().T
    print(num_summary)
    save_table(num_summary, "numerical_summary_statistics.csv")

    print("\nCategorical value counts:")
    cat_cols = ["customer_type", "category", "device_type"]
    for col in cat_cols:
        print(f"\n-- {col} --")
        print(tx[col].value_counts())

    valid_discounts = set(tx["discount_percentage"].unique()) <= set(DISCOUNT_LEVELS)
    valid_converted = set(tx["converted"].unique()) <= {0, 1}
    print(f"\ndiscount_percentage values only in {DISCOUNT_LEVELS}: {valid_discounts} "
          f"(observed: {sorted(tx['discount_percentage'].unique())})")
    print(f"converted values only in {{0,1}}: {valid_converted} "
          f"(observed: {sorted(tx['converted'].unique())})")

    if not valid_discounts:
        print("  *** FLAG: unexpected discount_percentage values present ***")
    if not valid_converted:
        print("  *** FLAG: converted is not strictly binary ***")


# --------------------------------------------------------------------------------------
# 2. TREATMENT ANALYSIS
# --------------------------------------------------------------------------------------
def treatment_analysis(tx):
    hr("2. TREATMENT ANALYSIS")

    n = len(tx)
    g = tx.groupby("discount_percentage")
    summary = pd.DataFrame({
        "n_transactions": g.size(),
        "pct_of_dataset": (g.size() / n * 100).round(2),
        "conversion_rate": g["converted"].mean().round(4),
        "avg_cart_value": g["cart_value"].mean().round(2),
        "avg_margin_pct": g["margin_percentage"].mean().round(2),
    })
    print(summary)
    save_table(summary, "treatment_analysis.csv")
    return summary


# --------------------------------------------------------------------------------------
# 3. CUSTOMER ANALYSIS
# --------------------------------------------------------------------------------------
def customer_analysis(tx):
    hr("3. CUSTOMER ANALYSIS")

    print("\n-- Conversion rate by customer_type --")
    t1 = tx.groupby("customer_type")["converted"].agg(["mean", "count"]).rename(
        columns={"mean": "conversion_rate", "count": "n"})
    print(t1)

    print("\n-- Conversion rate by previous_purchases (binned) --")
    bins_pp = [-1, 0, 1, 2, 5, 10, np.inf]
    labels_pp = ["0", "1", "2", "3-5", "6-10", "10+"]
    tx["_pp_bin"] = pd.cut(tx["previous_purchases"], bins=bins_pp, labels=labels_pp)
    t2 = tx.groupby("_pp_bin", observed=True)["converted"].agg(["mean", "count"]).rename(
        columns={"mean": "conversion_rate", "count": "n"})
    print(t2)

    print("\n-- Conversion rate by previous_abandons (binned) --")
    bins_pa = [-1, 0, 1, 2, 5, np.inf]
    labels_pa = ["0", "1", "2", "3-5", "5+"]
    tx["_pa_bin"] = pd.cut(tx["previous_abandons"], bins=bins_pa, labels=labels_pa)
    t3 = tx.groupby("_pa_bin", observed=True)["converted"].agg(["mean", "count"]).rename(
        columns={"mean": "conversion_rate", "count": "n"})
    print(t3)

    print("\n-- Conversion rate by device_type --")
    t4 = tx.groupby("device_type")["converted"].agg(["mean", "count"]).rename(
        columns={"mean": "conversion_rate", "count": "n"})
    print(t4)

    print("\n-- Conversion rate by category --")
    t5 = tx.groupby("category")["converted"].agg(["mean", "count"]).rename(
        columns={"mean": "conversion_rate", "count": "n"})
    print(t5)

    combined = pd.concat(
        {"customer_type": t1, "previous_purchases_bin": t2, "previous_abandons_bin": t3,
         "device_type": t4, "category": t5},
        names=["dimension", "value"]
    )
    save_table(combined, "customer_analysis.csv")

    tx.drop(columns=["_pp_bin", "_pa_bin"], inplace=True)
    return combined


# --------------------------------------------------------------------------------------
# 4. CHECKOUT BEHAVIOR
# --------------------------------------------------------------------------------------
def checkout_behavior(tx):
    hr("4. CHECKOUT BEHAVIOR")

    print("\n-- time_on_checkout_seconds: converted vs not --")
    print(tx.groupby("converted")["time_on_checkout_seconds"].describe()[["mean", "50%", "std"]])

    print("\n-- payment_attempts vs conversion --")
    t_pa = tx.groupby("payment_attempts")["converted"].agg(["mean", "count"]).rename(
        columns={"mean": "conversion_rate", "count": "n"})
    print(t_pa)

    print("\n-- pages_viewed (binned) vs conversion --")
    bins_pv = [0, 1, 2, 3, 5, np.inf]
    labels_pv = ["1", "2", "3", "4-5", "6+"]
    tx["_pv_bin"] = pd.cut(tx["pages_viewed"], bins=bins_pv, labels=labels_pv)
    t_pv = tx.groupby("_pv_bin", observed=True)["converted"].agg(["mean", "count"]).rename(
        columns={"mean": "conversion_rate", "count": "n"})
    print(t_pv)
    tx.drop(columns=["_pv_bin"], inplace=True)

    print("\n-- device_type vs conversion (recap) --")
    t_dev = tx.groupby("device_type")["converted"].agg(["mean", "count"]).rename(
        columns={"mean": "conversion_rate", "count": "n"})
    print(t_dev)

    combined = pd.concat(
        {"payment_attempts": t_pa, "pages_viewed_bin": t_pv, "device_type": t_dev},
        names=["dimension", "value"]
    )
    save_table(combined, "checkout_behavior.csv")
    return combined


# --------------------------------------------------------------------------------------
# 5. TRUE UPLIFT ANALYSIS (ground truth — evaluation only)
# --------------------------------------------------------------------------------------
def true_uplift_analysis(gt):
    hr("5. TRUE UPLIFT ANALYSIS  [GROUND TRUTH — EVALUATION ONLY, NOT ML FEATURES]")

    gt = gt.copy()
    # Recompute independently as a cross-check against the generator's own
    # true_incremental_lift_* columns (if present).
    gt["true_lift_5"] = gt["probability_5_discount"] - gt["probability_no_discount"]
    gt["true_lift_10"] = gt["probability_10_discount"] - gt["probability_no_discount"]
    gt["true_lift_15"] = gt["probability_15_discount"] - gt["probability_no_discount"]

    if "true_incremental_lift_5" in gt.columns:
        diff = (gt["true_lift_5"] - gt["true_incremental_lift_5"]).abs().max()
        print(f"Cross-check vs generator's true_incremental_lift_5 (max abs diff, "
              f"allowing for rounding): {diff:.4f}")

    lift_cols = ["true_lift_5", "true_lift_10", "true_lift_15"]
    stats = gt[lift_cols].agg(["mean", "median", "std", "min", "max"]).T
    print("\nTrue uplift summary statistics:")
    print(stats)
    save_table(stats, "true_uplift_summary.csv")

    if "segment" in gt.columns:
        print("\n-- True uplift by latent customer segment --")
        seg_stats = gt.groupby("segment")[lift_cols].mean().round(4)
        seg_stats["n"] = gt.groupby("segment").size()
        print(seg_stats)
        save_table(seg_stats, "true_uplift_by_segment.csv")
    else:
        seg_stats = None
        print("No 'segment' column found in ground_truth.csv — skipping segment breakdown.")

    return gt, stats, seg_stats


# --------------------------------------------------------------------------------------
# 6. ECONOMIC ANALYSIS
# --------------------------------------------------------------------------------------
def economic_analysis(tx, gt):
    hr("6. ECONOMIC ANALYSIS")

    merged = tx.merge(
        gt[["transaction_id", "probability_no_discount", "probability_5_discount",
            "probability_10_discount", "probability_15_discount", "segment"]],
        on="transaction_id", how="left", validate="one_to_one"
    )

    V = merged["cart_value"]
    m = merged["margin_percentage"] / 100.0
    cost = V * (1 - m)

    prob_map = {0: "probability_no_discount", 5: "probability_5_discount",
                10: "probability_10_discount", 15: "probability_15_discount"}

    for d in DISCOUNT_LEVELS:
        price_after_discount = V * (1 - d / 100.0)
        profit_if_converted = price_after_discount - cost
        p_convert = merged[prob_map[d]]
        merged[f"expected_profit_{d}"] = p_convert * profit_if_converted

    profit_cols = [f"expected_profit_{d}" for d in DISCOUNT_LEVELS]
    merged["true_max_expected_profit"] = merged[profit_cols].max(axis=1)

    def pick_optimal(row):
        vals = {d: row[f"expected_profit_{d}"] for d in DISCOUNT_LEVELS}
        return max(vals, key=vals.get)

    merged["true_optimal_discount"] = merged.apply(pick_optimal, axis=1)

    print("Expected profit summary by discount level (across all transactions):")
    print(merged[profit_cols].describe().T[["mean", "std", "min", "max"]])

    econ_summary = merged[profit_cols + ["true_max_expected_profit"]].describe().T
    save_table(econ_summary, "expected_profit_summary.csv")

    print(f"\nAverage true_max_expected_profit per transaction: "
          f"{merged['true_max_expected_profit'].mean():.2f}")

    keep_cols = ["transaction_id", "customer_id", "category", "segment", "cart_value",
                 "margin_percentage"] + profit_cols + \
        ["true_optimal_discount", "true_max_expected_profit"]
    econ_df = merged[keep_cols]
    save_table(econ_df.head(2000), "economic_analysis_sample.csv")

    return merged, econ_df


# --------------------------------------------------------------------------------------
# 7. OPTIMAL DISCOUNT ANALYSIS
# --------------------------------------------------------------------------------------
def optimal_discount_analysis(merged):
    hr("7. OPTIMAL DISCOUNT ANALYSIS")

    dist = merged["true_optimal_discount"].value_counts(normalize=True).sort_index() * 100
    dist = dist.reindex(DISCOUNT_LEVELS, fill_value=0.0)
    print("Percentage of transactions for which each discount is profit-maximizing:")
    print(dist.round(2).astype(str) + " %")
    save_table(dist.rename("pct_optimal").to_frame(), "optimal_discount_distribution.csv")

    max_share = dist.max()
    dominant_discount = dist.idxmax()
    if max_share >= 90:
        print(f"\n  *** FLAG: discount {dominant_discount}% is optimal for {max_share:.1f}% "
              f"of transactions. This is a near-trivial/degenerate dynamic-pricing problem — "
              f"a constant-discount policy would already be close to optimal, leaving little "
              f"for an uplift/pricing model to learn. ***")
    elif max_share >= 70:
        print(f"\n  *** CAUTION: discount {dominant_discount}% dominates ({max_share:.1f}% of "
              f"transactions). The problem still has some heterogeneity, but is skewed toward "
              f"a single action. Worth reviewing margin/price-sensitivity distributions. ***")
    else:
        print(f"\n  No single discount dominates (top share: {dominant_discount}% at "
              f"{max_share:.1f}%). Meaningful variation in the optimal action — good sign for "
              f"a non-trivial dynamic-pricing problem.")

    if "segment" in merged.columns:
        print("\n-- Optimal discount distribution by latent segment --")
        seg_dist = (
            merged.groupby("segment")["true_optimal_discount"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0) * 100
        )
        seg_dist = seg_dist.reindex(columns=DISCOUNT_LEVELS, fill_value=0.0).round(2)
        print(seg_dist)
        save_table(seg_dist, "optimal_discount_by_segment.csv")
    else:
        seg_dist = None

    print("\n-- Optimal discount distribution by category --")
    cat_dist = (
        merged.groupby("category")["true_optimal_discount"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0) * 100
    )
    cat_dist = cat_dist.reindex(columns=DISCOUNT_LEVELS, fill_value=0.0).round(2)
    print(cat_dist)
    save_table(cat_dist, "optimal_discount_by_category.csv")

    return dist, seg_dist, cat_dist


# --------------------------------------------------------------------------------------
# 8. CONVERSION VS PROFIT — illustrative examples
# --------------------------------------------------------------------------------------
def conversion_vs_profit_examples(merged):
    hr("8. CONVERSION VS PROFIT — ILLUSTRATIVE EXAMPLES (actual rows)")

    prob_cols = {0: "probability_no_discount", 5: "probability_5_discount",
                 10: "probability_10_discount", 15: "probability_15_discount"}
    profit_cols = {d: f"expected_profit_{d}" for d in DISCOUNT_LEVELS}

    m = merged.copy()
    m["_highest_conv_discount"] = m[list(prob_cols.values())].idxmax(axis=1).map(
        {v: k for k, v in prob_cols.items()})

    cols_to_show = (["transaction_id", "category", "cart_value", "margin_percentage"]
                     + list(prob_cols.values()) + list(profit_cols.values())
                     + ["true_optimal_discount"])

    print("\n-- Case A: highest discount (15%) has the highest conversion probability "
          "but is NOT the profit-optimal choice --")
    case_a = m[(m["_highest_conv_discount"] == 15) & (m["true_optimal_discount"] != 15)]
    print(f"({len(case_a):,} such transactions, {len(case_a)/len(m)*100:.1f}% of dataset)")
    print(case_a[cols_to_show].head(5).to_string(index=False))

    print("\n-- Case B: a LOWER discount produces higher expected profit than 15% --")
    case_b = m[(m["expected_profit_5"] > m["expected_profit_15"]) |
               (m["expected_profit_10"] > m["expected_profit_15"])]
    print(f"({len(case_b):,} such transactions, {len(case_b)/len(m)*100:.1f}% of dataset)")
    print(case_b[cols_to_show].head(5).to_string(index=False))

    print("\n-- Case C: 0% discount is optimal despite ANY discount increasing conversion --")
    case_c = m[
        (m["true_optimal_discount"] == 0)
        & (m["probability_15_discount"] > m["probability_no_discount"])
    ]
    print(f"({len(case_c):,} such transactions, {len(case_c)/len(m)*100:.1f}% of dataset)")
    print(case_c[cols_to_show].head(5).to_string(index=False))

    examples = pd.concat(
        {"highest_conv_not_optimal": case_a[cols_to_show].head(20),
         "lower_discount_higher_profit": case_b[cols_to_show].head(20),
         "zero_discount_optimal_despite_lift": case_c[cols_to_show].head(20)},
        names=["case"]
    )
    save_table(examples, "conversion_vs_profit_examples.csv")
    return case_a, case_b, case_c


# --------------------------------------------------------------------------------------
# 9. VISUALIZATIONS
# --------------------------------------------------------------------------------------
def make_visualizations(tx, gt_lift, merged):
    hr("9. VISUALIZATIONS")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Treatment distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    tx["discount_percentage"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#2b6cb0")
    ax.set_title("Treatment Distribution (Discount % assigned)")
    ax.set_xlabel("Discount %")
    ax.set_ylabel("Number of transactions")
    save_fig(fig, "01_treatment_distribution.png")

    # Conversion rate by discount
    fig, ax = plt.subplots(figsize=(6, 4))
    tx.groupby("discount_percentage")["converted"].mean().plot(kind="bar", ax=ax, color="#2f855a")
    ax.set_title("Conversion Rate by Discount Level")
    ax.set_xlabel("Discount %")
    ax.set_ylabel("Conversion rate")
    save_fig(fig, "02_conversion_rate_by_discount.png")

    # True uplift distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label in [("true_lift_5", "5%"), ("true_lift_10", "10%"), ("true_lift_15", "15%")]:
        sns.kdeplot(gt_lift[col], label=label, ax=ax)
    ax.set_title("True Uplift Distribution (vs. 0% discount)")
    ax.set_xlabel("Probability uplift")
    ax.legend(title="Discount")
    save_fig(fig, "03_true_uplift_distribution.png")

    # Expected profit by discount
    fig, ax = plt.subplots(figsize=(6, 4))
    profit_cols = [f"expected_profit_{d}" for d in DISCOUNT_LEVELS]
    merged[profit_cols].mean().plot(kind="bar", ax=ax, color="#b7791f")
    ax.set_xticklabels([f"{d}%" for d in DISCOUNT_LEVELS], rotation=0)
    ax.set_title("Average Expected Profit by Discount Level")
    ax.set_ylabel("Expected profit (currency units)")
    save_fig(fig, "04_expected_profit_by_discount.png")

    # True optimal discount distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    order = DISCOUNT_LEVELS
    merged["true_optimal_discount"].value_counts(normalize=True).reindex(order, fill_value=0)\
        .mul(100).plot(kind="bar", ax=ax, color="#805ad5")
    ax.set_xticklabels([f"{d}%" for d in order], rotation=0)
    ax.set_title("True Profit-Maximizing Discount Distribution")
    ax.set_ylabel("% of transactions")
    save_fig(fig, "05_true_optimal_discount_distribution.png")

    # Cart value distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(tx["cart_value"].clip(upper=tx["cart_value"].quantile(0.99)), bins=60, ax=ax,
                 color="#3182ce")
    ax.set_title("Cart Value Distribution (clipped at 99th pct)")
    save_fig(fig, "06_cart_value_distribution.png")

    # Margin distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=tx, x="margin_percentage", hue="category", bins=40, ax=ax,
                 element="step", stat="density", common_norm=False)
    ax.set_title("Margin % Distribution by Category")
    save_fig(fig, "07_margin_distribution_by_category.png")

    # Conversion vs checkout time
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_df = tx.copy()
    plot_df["checkout_time_clipped"] = plot_df["time_on_checkout_seconds"].clip(
        upper=plot_df["time_on_checkout_seconds"].quantile(0.99))
    bins = pd.qcut(plot_df["checkout_time_clipped"], 15, duplicates="drop")
    conv_by_time = plot_df.groupby(bins, observed=True)["converted"].mean()
    ax.plot(range(len(conv_by_time)), conv_by_time.values, marker="o", color="#c53030")
    ax.set_title("Conversion Rate vs Checkout Time (15 quantile bins)")
    ax.set_xlabel("Checkout time bin (low -> high)")
    ax.set_ylabel("Conversion rate")
    save_fig(fig, "08_conversion_vs_checkout_time.png")


# --------------------------------------------------------------------------------------
# CONCLUSION
# --------------------------------------------------------------------------------------
def print_conclusion(tx, gt_lift_stats, opt_dist, econ_merged):
    hr("CONCLUSION")

    conv_rate = tx["converted"].mean()
    reasonable_conv = 0.02 < conv_rate < 0.9

    mean_lift_5 = gt_lift_stats.loc["true_lift_5", "mean"]
    mean_lift_10 = gt_lift_stats.loc["true_lift_10", "mean"]
    mean_lift_15 = gt_lift_stats.loc["true_lift_15", "mean"]
    std_lift_15 = gt_lift_stats.loc["true_lift_15", "std"]
    has_heterogeneity = std_lift_15 > 0.02  # non-trivial spread in treatment effect

    diminishing = (mean_lift_10 - mean_lift_5) > (mean_lift_15 - mean_lift_10)

    max_share = opt_dist.max()
    meaningful_variation = max_share < 90

    print(f"1. Statistically reasonable data?\n"
          f"   Overall conversion rate = {conv_rate:.3f}. "
          f"{'Looks reasonable (not degenerate).' if reasonable_conv else 'FLAG: conversion rate looks degenerate.'}")

    print(f"\n2. Meaningful heterogeneous treatment effect (HTE)?\n"
          f"   Std dev of true_lift_15 across transactions = {std_lift_15:.4f}. "
          f"{'Yes — meaningful spread, HTE exists.' if has_heterogeneity else 'FLAG: little spread, HTE may be too weak to learn.'}")

    print(f"\n3. Is discount response diminishing (concave)?\n"
          f"   Mean lift 0->5: {mean_lift_5:.4f}, 5->10: {mean_lift_10 - mean_lift_5:.4f}, "
          f"10->15: {mean_lift_15 - mean_lift_10:.4f}. "
          f"{'Yes, diminishing marginal returns observed.' if diminishing else 'FLAG: marginal returns are not diminishing as expected.'}")

    print(f"\n4. Meaningful variation in the economically optimal discount?\n"
          f"   Most common optimal discount covers {max_share:.1f}% of transactions. "
          f"{'Yes, real variation exists.' if meaningful_variation else 'FLAG: one discount dominates — problem may be close to trivial.'}")

    suitable = reasonable_conv and has_heterogeneity and meaningful_variation
    print(f"\n5. Suitable for proceeding to uplift modeling?\n"
          f"   {'YES — the data shows a non-degenerate conversion rate, real treatment-effect heterogeneity, and meaningful variation in the profit-optimal discount. Proceeding to uplift modeling is reasonable.' if suitable else 'CAUTION — one or more checks above raised a flag. Review those sections before investing in uplift modeling, since the problem may be simpler (or more degenerate) than intended.'}")

    hr()


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tx, gt = load_data()

    basic_validation(tx, gt)
    treatment_analysis(tx)
    customer_analysis(tx)
    checkout_behavior(tx)
    gt_lift, gt_lift_stats, seg_stats = true_uplift_analysis(gt)
    merged, econ_df = economic_analysis(tx, gt)
    opt_dist, seg_dist, cat_dist = optimal_discount_analysis(merged)
    conversion_vs_profit_examples(merged)
    make_visualizations(tx, gt_lift, merged)

    print_conclusion(tx, gt_lift_stats, opt_dist, merged)

    print(f"\nAll tables and plots saved under: {OUT_DIR}/")


if __name__ == "__main__":
    main()