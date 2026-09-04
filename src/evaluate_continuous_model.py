"""
src/evaluate_continuous_model.py

Rigorous evaluation of the Stage-2 continuous discount-response model,
including counterfactual evaluation against the HIDDEN ground truth.

Ground truth is loaded here and ONLY here, strictly after predictions exist --
same hygiene as src/uplift_model.py's evaluate_uplift().

KEY POINT ON THE GROUND TRUTH
-----------------------------
The ground-truth CSVs store base_logit_no_discount, price_sensitivity, and
previous_abandons_at_eval. Per metadata.json those three fields fully determine
the generator's true_conversion_probability() at ANY discount in [0,100]. So we
can score the model at 2.5 / 7.5 / 15 / 25 / 33.33 / 57.91 / ... and not just at
the 12 stored true_prob_d* columns. We reimplement that function here
(evaluation only, never a feature).

KEY POINT ON THE ERROR FLOOR
----------------------------
base_logit_no_discount contains that session's own N(0, 0.45) noise draw, and
price_sensitivity is latent and never observable. So per-row error against an
individual true probability CANNOT go to zero -- the best any observable-feature
model can estimate is E[p | X_obs, d]. This script therefore reports:
  (a) raw per-row MAE/RMSE/bias                       -- includes the floor
  (b) a Monte-Carlo estimate of that irreducible floor -- for context
  (c) BINNED counterfactual comparisons                -- averaging cancels the
      noise and exposes genuine systematic bias
Read (c) as the real verdict on correctness; (a) alone understates the model.

Run from the project root:
    python src/evaluate_continuous_model.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import spearmanr

from src.continuous_response_model import (
    ARBITRARY_DISCOUNTS, DISCOUNT_COL, FORBIDDEN_COLUMNS, MODEL_PATH,
    REFERENCE_DISCOUNT_GRID, REQUIRED_INFERENCE_COLS, TARGET_COL,
    ContinuousResponseModel, resolve_data_dir,
)

warnings.filterwarnings("ignore")

OUT_DIR = "data/processed/uplift_continuous"

# Mirrors src/data_generation.py. Evaluation only.
MAX_DISCOUNT_EFFECT = 1.35
NOISE_SD = 0.45
GT_CLIP = (0.01, 0.99)

SEGMENT_COL = "segment"


def _hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def save_table(df, name, index=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=index)
    print(f"  [saved table] {path}")


def save_fig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved plot]  {path}")


# --------------------------------------------------------------------------------------
# GROUND-TRUTH RECONSTRUCTION (evaluation only)
# --------------------------------------------------------------------------------------
def true_conversion_probability(base_logit, price_sensitivity, previous_abandons, d):
    """Vectorized copy of data_generation.true_conversion_probability()."""
    d = np.clip(np.asarray(d, dtype=float), 0.0, 100.0)
    sat = np.log1p(d) / np.log1p(100.0)
    eff = (MAX_DISCOUNT_EFFECT * np.asarray(price_sensitivity, float) * sat
           + 0.12 * np.log1p(np.asarray(previous_abandons, float)) * sat)
    p = 1.0 / (1.0 + np.exp(-(np.asarray(base_logit, float) + eff)))
    return np.clip(p, *GT_CLIP)


def verify_gt_reconstruction(gt):
    """Confirm our reimplementation reproduces the STORED true_prob_d* columns
    before we trust it at off-grid discounts."""
    stored = [c for c in gt.columns if c.startswith("true_prob_d")]
    rows = []
    for col in stored:
        d = float(col.replace("true_prob_d", ""))
        recomputed = true_conversion_probability(
            gt["base_logit_no_discount"], gt["price_sensitivity"],
            gt["previous_abandons_at_eval"], d,
        )
        err = np.abs(recomputed - gt[col].to_numpy(float))
        rows.append({"discount": d, "max_abs_err": float(err.max()),
                     "mean_abs_err": float(err.mean())})
    table = pd.DataFrame(rows).sort_values("discount")
    print(table.to_string(index=False))
    worst = table["max_abs_err"].max()
    print(f"  worst deviation vs stored columns: {worst:.6f}")
    assert worst < 5e-4, (
        "Ground-truth reconstruction disagrees with the stored true_prob_d* "
        "columns. Do not trust off-grid ground truth until this is resolved."
    )
    print("  OK: reconstruction matches stored ground truth (rounding only).")
    return table


# --------------------------------------------------------------------------------------
# 1. FACTUAL PERFORMANCE
# --------------------------------------------------------------------------------------
def evaluate_factual(model, tx_test):
    y = tx_test[TARGET_COL].to_numpy().astype(int)
    p = model.predict_conversion_probability(tx_test, tx_test[DISCOUNT_COL].to_numpy())
    overall = {
        "n_test_rows": int(len(tx_test)),
        "observed_conversion_rate": float(y.mean()),
        "mean_predicted_probability": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
    }
    print(pd.Series(overall).to_string())
    save_table(pd.DataFrame([overall]), "factual_metrics.csv")

    # Reliability table (10 equal-count bins of predicted probability).
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], 10, duplicates="drop")
    calib = (df.groupby("bin", observed=True)
               .agg(mean_predicted=("p", "mean"),
                    observed_rate=("y", "mean"),
                    n=("y", "size"))
               .reset_index(drop=True))
    calib["gap"] = calib["mean_predicted"] - calib["observed_rate"]
    print("\nReliability table:")
    print(calib.to_string(index=False))
    save_table(calib, "calibration_table.csv")
    ece = float((calib["n"] / calib["n"].sum() * calib["gap"].abs()).sum())
    print(f"  expected calibration error (ECE): {ece:.5f}")

    # Calibration ALONG the discount axis. Because treatment is randomized,
    # observed conversion within a discount bin is an unbiased estimate of the
    # true mean response there -- this validates the response curve WITHOUT
    # using any ground truth at all.
    dd = pd.DataFrame({"d": tx_test[DISCOUNT_COL].to_numpy(), "p": p, "y": y})
    dd["dbin"] = pd.cut(dd["d"], bins=np.arange(0, 105, 5), include_lowest=True)
    by_d = (dd.groupby("dbin", observed=True)
              .agg(mean_discount=("d", "mean"), mean_predicted=("p", "mean"),
                   observed_rate=("y", "mean"), n=("y", "size"))
              .reset_index())
    by_d["gap"] = by_d["mean_predicted"] - by_d["observed_rate"]
    print("\nCalibration by discount bin (ground-truth-free check):")
    print(by_d.to_string(index=False))
    save_table(by_d, "calibration_by_discount_bin.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(calib["mean_predicted"], calib["observed_rate"], "o-", color="#2f855a")
    axes[0].plot([0, 1], [0, 1], "--", color="#c53030", lw=1)
    axes[0].set_xlabel("Mean predicted P(convert)")
    axes[0].set_ylabel("Observed conversion rate")
    axes[0].set_title("Reliability (factual test rows)")
    axes[1].plot(by_d["mean_discount"], by_d["mean_predicted"], "o-",
                 label="Predicted", color="#3182ce")
    axes[1].plot(by_d["mean_discount"], by_d["observed_rate"], "s--",
                 label="Observed", color="#c53030")
    axes[1].set_xlabel("Discount %")
    axes[1].set_ylabel("Conversion rate")
    axes[1].set_title("Response curve vs observed, by discount bin")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "01_calibration.png")

    overall["ece"] = ece
    overall["max_discount_bin_gap"] = float(by_d["gap"].abs().max())
    return overall, calib, by_d


# --------------------------------------------------------------------------------------
# 2. COUNTERFACTUAL EVALUATION VS HIDDEN GROUND TRUTH
# --------------------------------------------------------------------------------------
def evaluate_counterfactual(model, tx_test, gt_test, grid=None):
    grid = list(REFERENCE_DISCOUNT_GRID if grid is None else grid)

    merged = tx_test[["transaction_id"] + REQUIRED_INFERENCE_COLS].merge(
        gt_test[["transaction_id", SEGMENT_COL, "price_sensitivity",
                 "base_logit_no_discount", "previous_abandons_at_eval"]],
        on="transaction_id", how="inner", validate="one_to_one",
    )
    assert len(merged) == len(tx_test), "Ground-truth merge lost or duplicated rows."

    pred = model.predict_response_curve(merged[REQUIRED_INFERENCE_COLS], grid)
    true = np.column_stack([
        true_conversion_probability(
            merged["base_logit_no_discount"], merged["price_sensitivity"],
            merged["previous_abandons_at_eval"], d)
        for d in grid
    ])

    rows = []
    for j, d in enumerate(grid):
        e = pred[:, j] - true[:, j]
        rows.append({
            "discount": d,
            "mean_true": float(true[:, j].mean()),
            "mean_pred": float(pred[:, j].mean()),
            "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)),
            "corr": float(np.corrcoef(pred[:, j], true[:, j])[0, 1]),
        })
    per_point = pd.DataFrame(rows)
    print("Per-grid-point potential-outcome accuracy:")
    print(per_point.to_string(index=False))
    save_table(per_point, "counterfactual_metrics_by_discount.csv")

    # ---- uplift vs the d=0 baseline ----
    i0 = grid.index(0)
    pred_up = pred - pred[:, [i0]]
    true_up = true - true[:, [i0]]
    urows = []
    for j, d in enumerate(grid):
        if d == 0:
            continue
        e = pred_up[:, j] - true_up[:, j]
        sp = spearmanr(pred_up[:, j], true_up[:, j]).correlation
        urows.append({
            "discount": d,
            "mean_true_uplift": float(true_up[:, j].mean()),
            "mean_pred_uplift": float(pred_up[:, j].mean()),
            "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)),
            "spearman_targeting": float(sp),
        })
    uplift = pd.DataFrame(urows)
    print("\nUplift accuracy (vs 0% baseline). spearman_targeting = quality of "
          "the heterogeneity ranking:")
    print(uplift.to_string(index=False))
    save_table(uplift, "counterfactual_uplift_metrics.csv")

    # ---- irreducible noise floor (Monte Carlo, approximate) ----
    rng = np.random.default_rng(42)
    idx = rng.choice(len(merged), size=min(4000, len(merged)), replace=False)
    floors = []
    for j, d in enumerate(grid):
        base = merged["base_logit_no_discount"].to_numpy(float)[idx]
        ps = merged["price_sensitivity"].to_numpy(float)[idx]
        pa = merged["previous_abandons_at_eval"].to_numpy(float)[idx]
        centre = true_conversion_probability(base, ps, pa, d)
        draws = np.column_stack([
            true_conversion_probability(
                base + rng.normal(0, NOISE_SD, size=idx.size), ps, pa, d)
            for _ in range(20)
        ])
        floors.append({"discount": d,
                       "noise_floor_MAE": float(np.mean(np.abs(draws - centre[:, None]))),
                       "noise_floor_RMSE": float(np.sqrt(np.mean((draws - centre[:, None]) ** 2)))})
    floor = pd.DataFrame(floors)
    print("\nApproximate irreducible per-row error floor from the session noise "
          "draw alone (sd=0.45 on the logit scale). Per-row MAE at or near "
          "these values means the model is close to the achievable limit:")
    print(floor.to_string(index=False))
    save_table(floor, "irreducible_noise_floor.csv")

    # ---- BINNED counterfactual comparison: the real verdict ----
    # Bin by predicted p at d=0 (an observable-side statistic), then average.
    # Averaging cancels the independent per-session noise, so residual gaps are
    # genuine systematic bias rather than irreducible randomness.
    binner = pd.qcut(pred[:, i0], 20, duplicates="drop")
    frames = []
    for j, d in enumerate(grid):
        g = (pd.DataFrame({"bin": binner, "pred": pred[:, j], "true": true[:, j]})
               .groupby("bin", observed=True)
               .agg(mean_pred=("pred", "mean"), mean_true=("true", "mean"),
                    n=("true", "size"))
               .reset_index(drop=True))
        g["discount"] = d
        g["gap"] = g["mean_pred"] - g["mean_true"]
        frames.append(g)
    binned = pd.concat(frames, ignore_index=True)
    summary = (binned.groupby("discount")
                     .agg(binned_MAE=("gap", lambda s: float(np.mean(np.abs(s)))),
                          binned_max_gap=("gap", lambda s: float(np.max(np.abs(s)))),
                          binned_bias=("gap", "mean"))
                     .reset_index())
    print("\nBINNED counterfactual accuracy (20 bins of predicted p at d=0). "
          "This is the noise-cancelled read on systematic bias:")
    print(summary.to_string(index=False))
    save_table(binned, "counterfactual_binned_detail.csv")
    save_table(summary, "counterfactual_binned_summary.csv")

    # ---- by latent segment (reporting only) ----
    seg_rows = []
    for seg, sidx in merged.groupby(SEGMENT_COL).groups.items():
        pos = merged.index.get_indexer(sidx)
        for j, d in enumerate(grid):
            e = pred[pos, j] - true[pos, j]
            seg_rows.append({"segment": seg, "discount": d, "n": len(pos),
                             "mean_true": float(true[pos, j].mean()),
                             "mean_pred": float(pred[pos, j].mean()),
                             "MAE": float(np.mean(np.abs(e))),
                             "bias": float(np.mean(e))})
    by_segment = pd.DataFrame(seg_rows)
    print("\nMean true vs predicted curve by LATENT segment (segment is "
          "evaluation-only and was never a feature -- this shows how much "
          "unobserved heterogeneity the observable proxies recover):")
    piv = by_segment.pivot(index="discount", columns="segment",
                           values=["mean_true", "mean_pred"])
    print(piv.to_string())
    save_table(by_segment, "counterfactual_metrics_by_segment.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for seg in sorted(by_segment["segment"].unique()):
        s = by_segment[by_segment["segment"] == seg].sort_values("discount")
        line, = ax.plot(s["discount"], s["mean_true"], "-", label=f"{seg} (true)")
        ax.plot(s["discount"], s["mean_pred"], "--", color=line.get_color(),
                label=f"{seg} (pred)")
    ax.set_xlabel("Discount %")
    ax.set_ylabel("Mean P(convert)")
    ax.set_title("True vs predicted response curves by latent segment")
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, "02_curves_by_segment.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, d in zip(axes.flat, [0, 10, 25, 50, 75, 100]):
        j = int(np.argmin(np.abs(np.asarray(grid, float) - d)))
        ax.scatter(true[:, j], pred[:, j], s=6, alpha=0.12, color="#3182ce")
        ax.plot([0, 1], [0, 1], "--", color="#c53030", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"d = {grid[j]}%")
        ax.set_xlabel("True P(convert)"); ax.set_ylabel("Predicted P(convert)")
    fig.suptitle("Predicted vs true potential outcomes (spread reflects the "
                 "latent-sensitivity + session-noise floor)")
    fig.tight_layout()
    save_fig(fig, "03_pred_vs_true_scatter.png")

    return per_point, uplift, floor, summary, by_segment, merged, pred, true


# --------------------------------------------------------------------------------------
# 3. SMOOTHNESS / MONOTONICITY / DIMINISHING RETURNS
# --------------------------------------------------------------------------------------
def evaluate_curve_shape(model, tx_test, n_sample=500, step=0.5):
    dense = np.arange(0.0, 100.0 + step, step)
    sample = tx_test.sample(min(n_sample, len(tx_test)), random_state=42)
    curves = model.predict_response_curve(sample[REQUIRED_INFERENCE_COLS], dense)

    d1 = np.diff(curves, axis=1)
    d2 = np.diff(curves, n=2, axis=1)
    monotone = np.mean(np.all(d1 >= -1e-9, axis=1))
    near_monotone = np.mean(np.all(d1 >= -1e-4, axis=1))

    i20 = int(np.argmin(np.abs(dense - 20)))
    i70 = int(np.argmin(np.abs(dense - 70)))
    gain_low = curves[:, i20] - curves[:, 0]
    gain_high = curves[:, -1] - curves[:, i70]

    stats = {
        "n_customers_sampled": int(len(sample)),
        "dense_grid_step": step,
        "fraction_strictly_non_decreasing": float(monotone),
        "fraction_non_decreasing_tol_1e-4": float(near_monotone),
        "mean_abs_second_difference": float(np.mean(np.abs(d2))),
        "max_abs_second_difference": float(np.max(np.abs(d2))),
        "max_abs_first_difference_per_0.5pct": float(np.max(np.abs(d1))),
        "fraction_with_diminishing_returns": float(np.mean(gain_low > gain_high)),
        "mean_gain_0_to_20": float(gain_low.mean()),
        "mean_gain_70_to_100": float(gain_high.mean()),
    }
    print(pd.Series(stats).to_string())
    save_table(pd.DataFrame([stats]), "curve_shape_metrics.csv")

    # Labelled BASELINE: monotone-constrained GBM with d as a plain feature.
    # Included to document WHY a tree model was rejected on smoothness, not as
    # a candidate architecture.
    baseline_stats = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder
        from src.continuous_response_model import (CATEGORICAL_FEATURES,
                                                   NUMERICAL_FEATURES)
        data_dir = resolve_data_dir()
        tx_train = pd.read_csv(os.path.join(data_dir, "transactions_train.csv"))
        cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [DISCOUNT_COL]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr = tx_train[cols].copy()
        Xte = sample[cols].copy()
        Xtr[CATEGORICAL_FEATURES] = enc.fit_transform(Xtr[CATEGORICAL_FEATURES].astype(str))
        Xte[CATEGORICAL_FEATURES] = enc.transform(Xte[CATEGORICAL_FEATURES].astype(str))
        mono = [0] * len(cols)
        mono[cols.index(DISCOUNT_COL)] = 1
        gbm = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, monotonic_cst=mono, random_state=42)
        gbm.fit(Xtr, tx_train[TARGET_COL].to_numpy().astype(int))

        bc = np.empty((len(Xte), dense.size))
        for j, d in enumerate(dense):
            tmp = Xte.copy()
            tmp[DISCOUNT_COL] = d
            bc[:, j] = gbm.predict_proba(tmp)[:, 1]
        bd2 = np.diff(bc, n=2, axis=1)
        bd1 = np.diff(bc, axis=1)
        baseline_stats = {
            "model": "monotone HistGradientBoosting (BASELINE, rejected)",
            "mean_abs_second_difference": float(np.mean(np.abs(bd2))),
            "max_abs_second_difference": float(np.max(np.abs(bd2))),
            "fraction_of_grid_steps_exactly_flat": float(np.mean(np.abs(bd1) < 1e-12)),
        }
        print("\nSmoothness baseline for comparison:")
        print(pd.Series(baseline_stats).to_string())
        print("  Interpretation: a large 'fraction exactly flat' plus a large "
              "max second difference is the staircase signature -- flat "
              "plateaus separated by jumps at split points.")
        save_table(pd.DataFrame([baseline_stats]), "smoothness_baseline_gbm.csv")

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(min(4, len(sample))):
            line, = ax.plot(dense, curves[i], "-",
                            label=f"structured logistic #{i}" if i == 0 else None)
            ax.plot(dense, bc[i], "--", color=line.get_color(), alpha=0.7,
                    label="monotone GBM (baseline)" if i == 0 else None)
        ax.set_xlabel("Discount %"); ax.set_ylabel("Predicted P(convert)")
        ax.set_title("Smooth structured model (solid) vs staircase GBM (dashed)")
        ax.legend(fontsize=8)
        save_fig(fig, "04_smoothness_vs_gbm.png")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[baseline skipped: {type(exc).__name__}: {exc}]")

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(min(12, len(sample))):
        ax.plot(dense, curves[i], "-", lw=1)
    ax.set_xlabel("Discount %"); ax.set_ylabel("Predicted P(convert)")
    ax.set_title("Predicted response curves, 12 held-out customers "
                 "(0-100% at 0.5% steps)")
    save_fig(fig, "05_individual_curves.png")

    return stats, baseline_stats


# --------------------------------------------------------------------------------------
# 4. ARBITRARY-DISCOUNT TESTS
# --------------------------------------------------------------------------------------
def evaluate_arbitrary_discounts(model, merged):
    rows = []
    feats = merged[REQUIRED_INFERENCE_COLS]
    for d in ARBITRARY_DISCOUNTS:
        p = model.predict_conversion_probability(feats, d)
        t = true_conversion_probability(
            merged["base_logit_no_discount"], merged["price_sensitivity"],
            merged["previous_abandons_at_eval"], d)
        e = p - t
        rows.append({
            "discount": d,
            "n": int(len(p)),
            "all_finite": bool(np.all(np.isfinite(p))),
            "all_in_unit_interval": bool(np.all((p > 0) & (p < 1))),
            "mean_pred": float(p.mean()),
            "mean_true": float(t.mean()),
            "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)),
        })
    table = pd.DataFrame(rows)
    print("Off-grid discounts the model was never trained to treat specially, "
          "scored against recomputed ground truth:")
    print(table.to_string(index=False))
    save_table(table, "arbitrary_discount_results.csv")
    assert table["all_finite"].all() and table["all_in_unit_interval"].all(), \
        "Arbitrary-discount predictions were not valid probabilities."
    return table


# --------------------------------------------------------------------------------------
# 5. REPRESENTATIVE-CUSTOMER CURVES (mirrors response_curve_sanity_check.csv)
# --------------------------------------------------------------------------------------
def representative_curves(model, merged, n_per_segment=2):
    grid = REFERENCE_DISCOUNT_GRID
    picks = []
    for seg, g in merged.groupby(SEGMENT_COL):
        g = g.sort_values("price_sensitivity")
        idx = np.linspace(0, len(g) - 1, n_per_segment).astype(int)
        picks.append(g.iloc[idx])
    picked = pd.concat(picks, ignore_index=True)

    pred = model.predict_response_curve(picked[REQUIRED_INFERENCE_COLS], grid)
    rows = []
    for i, (_, r) in enumerate(picked.iterrows()):
        true = true_conversion_probability(
            r["base_logit_no_discount"], r["price_sensitivity"],
            r["previous_abandons_at_eval"], np.asarray(grid, float))
        row = {"segment": r[SEGMENT_COL], "customer_id": r.get("customer_id", ""),
               "price_sensitivity": round(float(r["price_sensitivity"]), 4)}
        for d, pv, tv in zip(grid, pred[i], np.atleast_1d(true)):
            row[f"pred_d{d}"] = round(float(pv), 4)
            row[f"true_d{d}"] = round(float(tv), 4)
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    save_table(table, "representative_customer_curves.csv")
    return table


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    _hr("LOAD MODEL AND DATA")
    model = ContinuousResponseModel.load(MODEL_PATH)
    print(f"Loaded {MODEL_PATH} (version {model.model_version})")
    data_dir = resolve_data_dir()
    tx_test = pd.read_csv(os.path.join(data_dir, "transactions_test.csv"))
    gt_test = pd.read_csv(os.path.join(data_dir, "ground_truth_test.csv"))
    print(f"test rows: {len(tx_test):,} | ground-truth rows: {len(gt_test):,}")

    leaked = set(tx_test.columns) & (FORBIDDEN_COLUMNS - {TARGET_COL})
    assert not leaked, f"Observable test frame unexpectedly contains {leaked}"
    print("Leakage guard: observable test frame carries no ground-truth column.")

    _hr("VERIFY GROUND-TRUTH RECONSTRUCTION")
    verify_gt_reconstruction(gt_test)

    _hr("1. FACTUAL PERFORMANCE (AUC / LOG LOSS / BRIER / CALIBRATION)")
    factual, calib, by_d = evaluate_factual(model, tx_test)

    _hr("2. COUNTERFACTUAL EVALUATION VS HIDDEN GROUND TRUTH")
    (per_point, uplift, floor, binned_summary,
     by_segment, merged, pred, true) = evaluate_counterfactual(model, tx_test, gt_test)

    _hr("3. CURVE SHAPE: SMOOTHNESS / MONOTONICITY / DIMINISHING RETURNS")
    shape, baseline = evaluate_curve_shape(model, tx_test)

    _hr("4. ARBITRARY-DISCOUNT PREDICTION TESTS")
    arbitrary = evaluate_arbitrary_discounts(model, merged)

    _hr("5. REPRESENTATIVE CUSTOMERS PER LATENT SEGMENT")
    reps = representative_curves(model, merged)

    _hr("SUMMARY")
    summary = {
        "model_path": MODEL_PATH,
        "model_version": model.model_version,
        "factual": factual,
        "counterfactual_overall": {
            "mean_MAE_across_grid": float(per_point["MAE"].mean()),
            "mean_RMSE_across_grid": float(per_point["RMSE"].mean()),
            "mean_bias_across_grid": float(per_point["bias"].mean()),
            "worst_point_MAE": float(per_point["MAE"].max()),
        },
        "counterfactual_binned": {
            "mean_binned_MAE": float(binned_summary["binned_MAE"].mean()),
            "worst_binned_max_gap": float(binned_summary["binned_max_gap"].max()),
        },
        "noise_floor_mean_MAE": float(floor["noise_floor_MAE"].mean()),
        "uplift": {
            "mean_MAE": float(uplift["MAE"].mean()),
            "mean_spearman_targeting": float(uplift["spearman_targeting"].mean()),
        },
        "curve_shape": shape,
        "smoothness_baseline_gbm": baseline,
        "arbitrary_discounts_all_valid": bool(
            arbitrary["all_in_unit_interval"].all()),
    }
    print(json.dumps(summary, indent=2, default=str))
    with open(os.path.join(OUT_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  [saved summary] {os.path.join(OUT_DIR, 'evaluation_summary.json')}")
    print(f"\nAll tables and plots are under {OUT_DIR}/")


if __name__ == "__main__":
    main()
