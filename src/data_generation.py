"""
src/data_generation.py

Synthetic checkout-session dataset generator for the Dynamic Pricing Agent
hackathon project (Razorpay "AI Growth & Agentic Commerce" track).

Produces:
    data/raw/synthetic_transactions.csv  - observable features + treatment + outcome
    data/raw/ground_truth.csv            - latent variables + counterfactual
                                            probabilities (evaluation only, never
                                            to be used as ML training features)

Run from the project root:
    python src/data_generation.py
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------
RANDOM_SEED = 42
N_TRANSACTIONS = 50_000
N_CUSTOMERS = 9_000          # inside the requested 8,000-10,000 range
N_DAYS = 90
DISCOUNT_LEVELS = [0, 5, 10, 15]
DISCOUNT_PROBS = [0.40, 0.30, 0.20, 0.10]  # treatment assignment - independent of latent vars

CATEGORIES = ["electronics", "fashion", "home", "beauty", "groceries", "accessories", "sports"]

# per-category cart-value (lognormal) and margin parameters
CATEGORY_PARAMS = {
    "electronics": dict(cart_mu=3400, cart_sigma=0.55, margin_low=8,  margin_high=15),
    "fashion":     dict(cart_mu=1300, cart_sigma=0.55, margin_low=30, margin_high=50),
    "home":        dict(cart_mu=2000, cart_sigma=0.60, margin_low=20, margin_high=35),
    "beauty":      dict(cart_mu=800,  cart_sigma=0.45, margin_low=35, margin_high=55),
    "groceries":   dict(cart_mu=650,  cart_sigma=0.35, margin_low=5,  margin_high=12),
    "accessories": dict(cart_mu=900,  cart_sigma=0.45, margin_low=25, margin_high=40),
    "sports":      dict(cart_mu=1500, cart_sigma=0.50, margin_low=20, margin_high=35),
}

DEVICE_TYPES = ["mobile", "desktop", "tablet"]
DEVICE_PROBS = [0.55, 0.35, 0.10]

SEGMENTS = ["ready_buyer", "price_sensitive", "hesitant", "low_intent"]
SEGMENT_PROBS = [0.23, 0.27, 0.28, 0.22]  # deliberately irregular, not exact quarters

# price_sensitivity ~ Beta(a, b) per segment - correlated with segment but overlapping,
# so segments are not trivially separable on this variable alone.
SEGMENT_SENSITIVITY_BETA = {
    "ready_buyer":     (2.0, 8.0),   # mean ~0.20 - discounts barely move them
    "price_sensitive": (8.0, 2.5),   # mean ~0.76 - discounts move them a lot
    "hesitant":        (4.0, 3.5),   # mean ~0.53 - moderate response
    "low_intent":      (2.0, 6.0),   # mean ~0.25 - price-aware but mostly just uninterested
}

# baseline logit intercept per segment (this is what drives "base_intent" before
# any discount is applied)
SEGMENT_BASE_INTENT = {
    "ready_buyer":     1.65,
    "price_sensitive": -0.85,
    "hesitant":        -0.55,
    "low_intent":      -2.35,
}

MAX_DISCOUNT_EFFECT = 1.35   # logit-scale ceiling of the discount effect at full price sensitivity
NOISE_SD = 0.45
VIP_CLV_THRESHOLD = 8000

OUT_DIR_RAW = "data/raw"
TRANSACTIONS_PATH = os.path.join(OUT_DIR_RAW, "synthetic_transactions.csv")
GROUND_TRUTH_PATH = os.path.join(OUT_DIR_RAW, "ground_truth.csv")

rng = np.random.default_rng(RANDOM_SEED)

# hourly checkout-activity weights: low overnight, ramps up through the morning,
# peaks at lunch and again in the evening
HOUR_WEIGHTS = np.array([
    0.5, 0.3, 0.2, 0.2, 0.2, 0.3,
    0.6, 1.0, 1.4, 1.6, 1.7, 1.9,
    2.3, 2.1, 1.8, 1.6, 1.7, 1.9,
    2.4, 2.6, 2.3, 1.8, 1.3, 0.8,
])
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()


# --------------------------------------------------------------------------------------
# CUSTOMER GENERATION
# --------------------------------------------------------------------------------------
def generate_customers(n_customers):
    """Latent, never-observed-downstream customer attributes: segment, price
    sensitivity, and soft device/category preferences."""
    customer_ids = [f"C{100000 + i}" for i in range(n_customers)]
    segments = rng.choice(SEGMENTS, size=n_customers, p=SEGMENT_PROBS)

    price_sensitivity = np.empty(n_customers)
    for seg in SEGMENTS:
        mask = segments == seg
        a, b = SEGMENT_SENSITIVITY_BETA[seg]
        price_sensitivity[mask] = rng.beta(a, b, size=mask.sum())
    # small extra independent noise so segments overlap on this variable
    price_sensitivity = np.clip(price_sensitivity + rng.normal(0, 0.04, n_customers), 0.01, 0.99)

    device_pref = rng.choice(DEVICE_TYPES, size=n_customers, p=DEVICE_PROBS)
    category_pref = rng.choice(CATEGORIES, size=n_customers)

    return pd.DataFrame({
        "customer_id": customer_ids,
        "segment": segments,
        "price_sensitivity": price_sensitivity,
        "device_pref": device_pref,
        "category_pref": category_pref,
    })


def assign_session_counts(n_customers, n_transactions):
    """Right-skewed number of sessions per customer, nudged so the total matches
    n_transactions exactly (guarantees the final row count)."""
    counts = 1 + rng.poisson(3.2, size=n_customers)
    counts = np.clip(counts, 1, 40)

    diff = n_transactions - counts.sum()
    order = rng.permutation(n_customers)
    i = 0
    while diff != 0:
        c = order[i % n_customers]
        if diff > 0:
            counts[c] += 1
            diff -= 1
        elif counts[c] > 1:
            counts[c] -= 1
            diff += 1
        i += 1
    return counts


def generate_customer_history(n_sessions_for_customer, n_days):
    """Sorted session timestamps for one customer across the observation window,
    using a realistic hour-of-day activity distribution."""
    days = rng.integers(0, n_days, size=n_sessions_for_customer)
    hours = rng.choice(24, size=n_sessions_for_customer, p=HOUR_WEIGHTS)
    minutes = rng.integers(0, 60, size=n_sessions_for_customer)
    seconds = rng.integers(0, 60, size=n_sessions_for_customer)
    base = datetime(2026, 1, 1)
    timestamps = [
        base + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(days, hours, minutes, seconds)
    ]
    return sorted(timestamps)


# --------------------------------------------------------------------------------------
# CONVERSION MODEL
# --------------------------------------------------------------------------------------
def discount_effect(discount_pct, price_sensitivity, previous_abandons):
    """Saturating (diminishing-returns) discount effect on the logit scale, scaled
    by the customer's latent price sensitivity. A small nonlinear interaction term
    means customers who have abandoned before respond a bit more to a discount."""
    saturating = np.log1p(discount_pct) / np.log1p(15)  # in [0, 1]
    base_effect = MAX_DISCOUNT_EFFECT * price_sensitivity * saturating
    interaction = 0.12 * np.log1p(previous_abandons) * saturating
    return base_effect + interaction


def session_base_logit(segment, previous_purchases, previous_abandons, days_since_last_purchase,
                        clv, cart_value, category, device_type, time_on_checkout,
                        payment_attempts, pages_viewed, hour, day_of_week, noise):
    """Every logit component EXCEPT the discount term, so the same base can be reused
    across all four counterfactual discount levels for a given session."""
    logit = SEGMENT_BASE_INTENT[segment]

    logit += 0.35 * np.log1p(previous_purchases)
    logit -= 0.40 * np.log1p(previous_abandons)

    if days_since_last_purchase is not None and days_since_last_purchase >= 0:
        logit -= 0.002 * min(days_since_last_purchase, 90)

    logit += 0.08 * np.log1p(clv / 1000.0)

    cat_mu = CATEGORY_PARAMS[category]["cart_mu"]
    logit -= 0.10 * (np.log(cart_value + 1) - np.log(cat_mu + 1))

    if device_type == "mobile":
        logit -= 0.10
    elif device_type == "tablet":
        logit -= 0.03

    logit -= 0.15 * (time_on_checkout / 120.0)
    logit -= 0.25 * max(0, payment_attempts - 1)
    logit -= 0.03 * max(0, pages_viewed - 3)

    if 18 <= hour <= 21:
        logit += 0.10
    if day_of_week >= 5:
        logit += 0.05

    logit += noise
    return logit


def calculate_conversion_probability(segment, price_sensitivity, previous_purchases, previous_abandons,
                                      days_since_last_purchase, clv, cart_value, category, device_type,
                                      time_on_checkout, payment_attempts, pages_viewed, hour, day_of_week):
    """Returns a dict {0:p, 5:p, 10:p, 15:p} - the counterfactual conversion
    probability under each discount level, sharing a single noise draw across all
    four (potential-outcomes framing: same unit, same shock, different treatment)."""
    noise = float(rng.normal(0, NOISE_SD))
    base_logit = session_base_logit(
        segment, previous_purchases, previous_abandons, days_since_last_purchase,
        clv, cart_value, category, device_type, time_on_checkout,
        payment_attempts, pages_viewed, hour, day_of_week, noise
    )
    probs = {}
    for d in DISCOUNT_LEVELS:
        eff = discount_effect(d, price_sensitivity, previous_abandons)
        p = 1.0 / (1.0 + np.exp(-(base_logit + eff)))
        probs[d] = float(np.clip(p, 0.02, 0.97))
    return probs


def assign_discount():
    """Randomized treatment assignment, independent of any customer/session feature."""
    return int(rng.choice(DISCOUNT_LEVELS, p=DISCOUNT_PROBS))


def generate_outcome(probs, assigned_discount):
    p_actual = probs[assigned_discount]
    converted = int(rng.random() < p_actual)
    return converted, p_actual


# --------------------------------------------------------------------------------------
# CHECKOUT-SESSION FEATURE SAMPLING (non-conversion-related)
# --------------------------------------------------------------------------------------
def sample_category(category_pref):
    return category_pref if rng.random() < 0.7 else rng.choice(CATEGORIES)


def sample_cart_and_margin(category):
    params = CATEGORY_PARAMS[category]
    cart_value = float(rng.lognormal(np.log(params["cart_mu"]), params["cart_sigma"]))
    cart_value = round(min(cart_value, params["cart_mu"] * 8), 2)
    margin_percentage = round(float(rng.uniform(params["margin_low"], params["margin_high"])), 2)
    items_count = int(np.clip(round(rng.normal(cart_value / 700.0, 1.0)), 1, 20))
    return cart_value, margin_percentage, items_count


def sample_device(device_pref):
    return device_pref if rng.random() < 0.8 else rng.choice(DEVICE_TYPES, p=DEVICE_PROBS)


def sample_checkout_behavior(segment):
    base_mean_time = 130 if segment in ("hesitant", "price_sensitive") else 90
    time_on_checkout = float(min(rng.lognormal(np.log(base_mean_time), 0.5), 1800))
    payment_attempts = int(1 + rng.binomial(2, 0.12))
    pages_viewed = int(min(1 + rng.poisson(2.5), 15))
    return round(time_on_checkout, 1), payment_attempts, pages_viewed


# --------------------------------------------------------------------------------------
# MAIN SESSION-GENERATION LOOP (sequential per customer to avoid target leakage)
# --------------------------------------------------------------------------------------
def generate_checkout_sessions(customers, session_counts):
    tx_rows = []
    gt_rows = []
    tmp_id_counter = 0

    for idx in range(len(customers)):
        cust = customers.iloc[idx]
        customer_id = cust["customer_id"]
        segment = cust["segment"]
        price_sensitivity = cust["price_sensitivity"]
        device_pref = cust["device_pref"]
        category_pref = cust["category_pref"]

        n_sessions = int(session_counts[idx])
        timestamps = generate_customer_history(n_sessions, N_DAYS)

        previous_purchases = 0
        previous_abandons = 0
        last_purchase_ts = None
        cumulative_clv = 0.0
        discount_history_sum = 0.0
        discount_history_n = 0

        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.weekday()

            days_since_last_purchase = (ts - last_purchase_ts).days if last_purchase_ts is not None else -1
            historical_discount_rate = (
                discount_history_sum / discount_history_n if discount_history_n > 0 else 5.0
            )
            customer_type = (
                "new" if previous_purchases == 0
                else "vip" if cumulative_clv > VIP_CLV_THRESHOLD
                else "returning"
            )

            category = sample_category(category_pref)
            cart_value, margin_percentage, items_count = sample_cart_and_margin(category)
            device_type = sample_device(device_pref)
            time_on_checkout, payment_attempts, pages_viewed = sample_checkout_behavior(segment)

            probs = calculate_conversion_probability(
                segment, price_sensitivity, previous_purchases, previous_abandons,
                days_since_last_purchase, cumulative_clv, cart_value, category, device_type,
                time_on_checkout, payment_attempts, pages_viewed, hour, day_of_week
            )
            assigned_discount = assign_discount()
            converted, _ = generate_outcome(probs, assigned_discount)

            tmp_id = tmp_id_counter
            tmp_id_counter += 1

            tx_rows.append({
                "transaction_id": tmp_id,
                "customer_id": customer_id,
                "timestamp": ts,
                "customer_type": customer_type,
                "previous_purchases": previous_purchases,
                "previous_abandons": previous_abandons,
                "days_since_last_purchase": days_since_last_purchase,
                "customer_lifetime_value": round(cumulative_clv, 2),
                "historical_discount_rate": round(historical_discount_rate, 2),
                "cart_value": cart_value,
                "items_count": items_count,
                "category": category,
                "margin_percentage": margin_percentage,
                "device_type": device_type,
                "time_on_checkout_seconds": time_on_checkout,
                "pages_viewed": pages_viewed,
                "payment_attempts": payment_attempts,
                "hour": hour,
                "day_of_week": day_of_week,
                "discount_percentage": assigned_discount,
                "converted": converted,
            })

            gt_rows.append({
                "transaction_id": tmp_id,
                "customer_id": customer_id,
                "segment": segment,
                "price_sensitivity": round(price_sensitivity, 4),
                "base_conversion_probability": probs[0],
                "probability_no_discount": probs[0],
                "probability_5_discount": probs[5],
                "probability_10_discount": probs[10],
                "probability_15_discount": probs[15],
                "true_incremental_lift_5": round(probs[5] - probs[0], 4),
                "true_incremental_lift_10": round(probs[10] - probs[0], 4),
                "true_incremental_lift_15": round(probs[15] - probs[0], 4),
            })

            # update history AFTER this session's features/outcome are recorded
            if converted:
                previous_purchases += 1
                cumulative_clv += cart_value
                last_purchase_ts = ts
            else:
                previous_abandons += 1
            discount_history_sum += assigned_discount
            discount_history_n += 1

    return pd.DataFrame(tx_rows), pd.DataFrame(gt_rows)


def finalize_and_sort(tx_df, gt_df):
    """Sort chronologically and replace temporary integer ids with clean, globally
    unique, time-ordered transaction ids."""
    tx_df = tx_df.sort_values("timestamp").reset_index(drop=True)
    id_map = {old: f"TXN{i:06d}" for i, old in enumerate(tx_df["transaction_id"])}
    tx_df["transaction_id"] = tx_df["transaction_id"].map(id_map)
    gt_df["transaction_id"] = gt_df["transaction_id"].map(id_map)
    gt_df = gt_df.set_index("transaction_id").loc[tx_df["transaction_id"]].reset_index()
    return tx_df, gt_df


# --------------------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------------------
def validate_dataset(tx_df, gt_df):
    print("=" * 72)
    print("VALIDATION REPORT")
    print("=" * 72)

    n_rows = len(tx_df)
    n_customers = tx_df["customer_id"].nunique()
    conv_rate = tx_df["converted"].mean()

    print(f"Rows: {n_rows}")
    print(f"Unique customers: {n_customers}")
    print(f"Overall conversion rate: {conv_rate:.3f}\n")

    print("Treatment distribution (discount_percentage):")
    print(tx_df["discount_percentage"].value_counts(normalize=True).sort_index(), "\n")

    print(f"Average cart value: {tx_df['cart_value'].mean():.2f}")
    print(f"Median cart value: {tx_df['cart_value'].median():.2f}")
    print(f"Average margin %: {tx_df['margin_percentage'].mean():.2f}\n")

    print("Conversion rate by discount level:")
    print(tx_df.groupby("discount_percentage")["converted"].mean(), "\n")

    print("Conversion rate by customer_type:")
    print(tx_df.groupby("customer_type")["converted"].mean(), "\n")

    print("Conversion rate by category:")
    print(tx_df.groupby("category")["converted"].mean(), "\n")

    print("Conversion rate by device_type:")
    print(tx_df.groupby("device_type")["converted"].mean(), "\n")

    missing = tx_df.isnull().sum()
    missing = missing[missing > 0]
    print("Missing values per column:")
    print(missing if len(missing) else "None", "\n")

    dup_ids = tx_df["transaction_id"].duplicated().sum()
    print(f"Duplicate transaction IDs: {dup_ids}\n")

    numeric_cols = tx_df.select_dtypes(include=[np.number]).columns
    print("Numeric column ranges (min / max / mean):")
    print(tx_df[numeric_cols].agg(["min", "max", "mean"]).T, "\n")

    # ---- hard assertions: fail loudly on violation ----
    assert n_rows == N_TRANSACTIONS, f"Expected {N_TRANSACTIONS} rows, got {n_rows}"
    assert dup_ids == 0, "Duplicate transaction_id values found"
    assert n_customers < n_rows, "customer_id values should repeat (returning customers)"
    assert set(tx_df["discount_percentage"].unique()) <= set(DISCOUNT_LEVELS), \
        "Invalid discount_percentage value present"
    assert set(tx_df["converted"].unique()) <= {0, 1}, "converted must be binary"

    non_negative_cols = ["cart_value", "items_count", "time_on_checkout_seconds",
                          "payment_attempts", "pages_viewed", "margin_percentage",
                          "previous_purchases", "previous_abandons", "customer_lifetime_value"]
    assert (tx_df[non_negative_cols] < 0).sum().sum() == 0, \
        "Negative values found in fields that must be non-negative"

    required_fields = ["customer_id", "timestamp", "cart_value", "discount_percentage", "converted"]
    assert tx_df[required_fields].isnull().sum().sum() == 0, \
        "Missing values found in required fields"

    prob_cols = ["base_conversion_probability", "probability_no_discount",
                 "probability_5_discount", "probability_10_discount", "probability_15_discount"]
    assert gt_df[prob_cols].min().min() >= 0.0 and gt_df[prob_cols].max().max() <= 1.0, \
        "Ground-truth probabilities out of [0, 1] range"
    assert conv_rate < 0.90, "Conversion rate suspiciously high for synthetic data (>=90%)"

    print("All validation checks passed.")
    print("=" * 72)


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR_RAW, exist_ok=True)

    customers = generate_customers(N_CUSTOMERS)
    session_counts = assign_session_counts(N_CUSTOMERS, N_TRANSACTIONS)

    tx_df, gt_df = generate_checkout_sessions(customers, session_counts)
    tx_df, gt_df = finalize_and_sort(tx_df, gt_df)

    validate_dataset(tx_df, gt_df)

    tx_df.to_csv(TRANSACTIONS_PATH, index=False)
    gt_df.to_csv(GROUND_TRUTH_PATH, index=False)

    print(f"\nSaved: {TRANSACTIONS_PATH}  ({len(tx_df)} rows)")
    print(f"Saved: {GROUND_TRUTH_PATH}  ({len(gt_df)} rows)")


if __name__ == "__main__":
    main()