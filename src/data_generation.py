"""
src/data_generation.py

Synthetic checkout-session dataset generator for the Dynamic Pricing Agent
hackathon project (Razorpay "AI Growth & Agentic Commerce" track).

Two modes, selected via --mode (default is unchanged from before: "discrete"):

  discrete (default, UNCHANGED from before):
    data/raw/synthetic_transactions.csv  - observable features + treatment + outcome
    data/raw/ground_truth.csv            - latent variables + counterfactual
                                            probabilities (evaluation only, never
                                            to be used as ML training features)
    Discount treatment is one of the fixed arms {0, 5, 10, 15}%.

  continuous (NEW - additive, never overwrites the discrete files above):
    data/raw/continuous/transactions_train.csv / transactions_test.csv
    data/raw/continuous/ground_truth_train.csv / ground_truth_test.csv
    data/raw/continuous/response_curve_sanity_check.csv
    data/raw/continuous/metadata.json
    Discount treatment is a continuous value in [0, 100]%, for a future
    continuous discount-response model. Reuses the same latent
    segment/price-sensitivity customer model and the same diminishing-returns
    mathematical family as the discrete mode, just re-normalized over the
    full 0-100% range instead of 0-15%, so the two datasets stay conceptually
    consistent and the discrete system remains fully rollback-safe.

Run from the project root:
    python src/data_generation.py                     # discrete only (as before)
    python src/data_generation.py --mode continuous    # new continuous dataset only
    python src/data_generation.py --mode both          # both
"""

import json
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
# CONTINUOUS-TREATMENT CONFIGURATION (NEW - additive, discrete config above is untouched)
# --------------------------------------------------------------------------------------
# Scaled up from the discrete run's 9,000 customers / 50,000 sessions (~5.6
# sessions/customer) to stay in the requested 50k-100k row range at a similar
# density (~6.7 sessions/customer).
CONTINUOUS_N_CUSTOMERS = 12_000
CONTINUOUS_N_TRANSACTIONS = 80_000
CONTINUOUS_TEST_SIZE = 0.20  # customer-level split

# Treatment assignment: mostly a smooth, right-skewed continuous draw across
# the full 0-100% range (most discounts modest, a long tail reaching toward
# 100%), with a small share landing on "round" discount presets a merchant
# might realistically configure. Both parts are independent of every
# customer/session feature, same as the discrete assign_discount(), so this
# stays a randomized-experiment-style design.
COMMON_ROUND_DISCOUNTS = [0, 5, 10, 15, 20, 25, 30, 40, 50]
ROUND_DISCOUNT_PROB = 0.15
# A pure Beta-skewed draw leaves the 80-100% region almost empty (a real,
# but very thin, tail) - a later model would have little to learn from up
# there. This uniform slice guarantees genuine training density across the
# ENTIRE 0-100% range (not just analytically-evaluable via the hidden
# ground-truth function), while the remaining ~75% keeps the realistic,
# mostly-modest-discount shape.
UNIFORM_MIX_PROB = 0.10
CONTINUOUS_DISCOUNT_BETA = (1.8, 4.5)  # Beta(a,b)*100 -> mean ~28%, long tail to 100%

# Fixed grid used only for reporting/plotting/sanity-checking the response
# curve - NOT a restriction on the treatment itself, which stays continuous.
REFERENCE_DISCOUNT_GRID = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# How the (previously unused-but-required-elsewhere) recent_discount_count /
# days_since_last_discount fields are computed - matches
# config/policy.yaml's discount_cooldown_days and how
# src/policy_engine.py's check_frequency_limit() already interprets them.
DISCOUNT_COOLDOWN_DAYS = 7

CONTINUOUS_OUT_DIR = os.path.join(OUT_DIR_RAW, "continuous")


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
# MAIN (discrete - UNCHANGED)
# --------------------------------------------------------------------------------------
def main():
    global rng
    rng = np.random.default_rng(RANDOM_SEED)  # fresh, independent stream every run

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


# ========================================================================================
# CONTINUOUS-TREATMENT DATASET (NEW)
#
# Everything below is additive: it reuses generate_customers(),
# assign_session_counts(), generate_customer_history(), session_base_logit(),
# sample_category()/sample_cart_and_margin()/sample_device()/
# sample_checkout_behavior(), and finalize_and_sort() UNCHANGED from the
# discrete section above. Only the discount treatment, the outcome model's
# discount term, and two extra history fields (recent_discount_count,
# days_since_last_discount - required by src/policy_engine.py's
# CheckoutDecisionRequest and demoed in src/checkout_pricing.py, but never
# actually produced by the discrete generator above) are new.
# ========================================================================================
def assign_continuous_discount():
    """Continuous treatment assignment in [0, 100]%, independent of every
    customer/session feature (same experimental-design property as the
    discrete assign_discount()). Three-way mixture:
      - 15% common 'round' presets a merchant might realistically configure
      - 10% uniform over the FULL [0,100] range, so the model gets genuine
        training density even in the rarely-realistic 80-100% region
      - 75% a right-skewed continuous draw (most discounts modest, a long
        tail toward 100%) - this is what makes the overall shape realistic
    """
    r = rng.random()
    if r < ROUND_DISCOUNT_PROB:
        return float(rng.choice(COMMON_ROUND_DISCOUNTS))
    if r < ROUND_DISCOUNT_PROB + UNIFORM_MIX_PROB:
        return float(rng.uniform(0.0, 100.0))
    a, b = CONTINUOUS_DISCOUNT_BETA
    return float(np.clip(rng.beta(a, b) * 100.0, 0.0, 100.0))


def continuous_discount_effect(discount_pct, price_sensitivity, previous_abandons):
    """Same diminishing-returns mathematical family as the discrete
    discount_effect(), re-normalized by log1p(100) instead of log1p(15) so the
    saturating curve is smooth and well-behaved across the FULL continuous
    0-100% range (still reaches the same MAX_DISCOUNT_EFFECT ceiling at full
    price sensitivity and 100% discount - consistent with the discrete model,
    not a different formula)."""
    discount_pct = np.clip(discount_pct, 0.0, 100.0)
    saturating = np.log1p(discount_pct) / np.log1p(100.0)  # in [0, 1], smooth & monotonic
    base_effect = MAX_DISCOUNT_EFFECT * price_sensitivity * saturating
    interaction = 0.12 * np.log1p(previous_abandons) * saturating
    return base_effect + interaction


def true_conversion_probability(base_logit_no_discount, price_sensitivity, previous_abandons, discount_pct):
    """Hidden ground-truth P(conversion) at ANY discount in [0, 100], given a
    session's already-realized non-discount logit (which already bakes in
    that session's own noise draw) and its latent price sensitivity. This is
    what lets later analysis ask true_P_conversion(customer, 47.8%) for any
    value at all - not just a fixed grid. Evaluation-only: never a training
    feature, and the ground-truth CSVs store exactly the three inputs this
    function needs (base_logit_no_discount, price_sensitivity,
    previous_abandons_at_eval) so this can be recomputed for arbitrary
    discounts later without re-running the generator."""
    eff = continuous_discount_effect(discount_pct, price_sensitivity, previous_abandons)
    p = 1.0 / (1.0 + np.exp(-(base_logit_no_discount + eff)))
    return float(np.clip(p, 0.01, 0.99))


def generate_continuous_checkout_sessions(customers, session_counts):
    """Same sequential, leakage-free, per-customer structure as
    generate_checkout_sessions() above, extended with a continuous discount
    treatment and two new rolling history fields."""
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
        discount_event_timestamps = []  # prior sessions where discount_percentage > 0

        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.weekday()

            days_since_last_purchase = (ts - last_purchase_ts).days if last_purchase_ts is not None else -1
            historical_discount_rate = (
                discount_history_sum / discount_history_n if discount_history_n > 0 else 5.0
            )
            days_since_last_discount = (
                (ts - discount_event_timestamps[-1]).days if discount_event_timestamps else -1
            )
            recent_discount_count = sum(
                1 for t in discount_event_timestamps if (ts - t).days < DISCOUNT_COOLDOWN_DAYS
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

            # non-discount logit, shared across the assigned discount AND every
            # reference-grid counterfactual below (same noise draw, per the
            # potential-outcomes framing already used in the discrete model)
            noise = float(rng.normal(0, NOISE_SD))
            base_logit_no_discount = session_base_logit(
                segment, previous_purchases, previous_abandons, days_since_last_purchase,
                cumulative_clv, cart_value, category, device_type, time_on_checkout,
                payment_attempts, pages_viewed, hour, day_of_week, noise
            )

            assigned_discount = assign_continuous_discount()
            p_actual = true_conversion_probability(
                base_logit_no_discount, price_sensitivity, previous_abandons, assigned_discount
            )
            converted = int(rng.random() < p_actual)

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
                "recent_discount_count": recent_discount_count,
                "days_since_last_discount": days_since_last_discount,
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
                "discount_percentage": round(assigned_discount, 4),
                "converted": converted,
            })

            ref_probs = {
                d: true_conversion_probability(base_logit_no_discount, price_sensitivity, previous_abandons, d)
                for d in REFERENCE_DISCOUNT_GRID
            }
            gt_row = {
                "transaction_id": tmp_id,
                "customer_id": customer_id,
                "segment": segment,
                "price_sensitivity": round(price_sensitivity, 4),
                # the three fields below fully determine
                # true_conversion_probability(customer, ANY discount) - see that
                # function's docstring
                "base_logit_no_discount": round(base_logit_no_discount, 6),
                "previous_abandons_at_eval": previous_abandons,
                "true_probability_at_assigned_discount": round(p_actual, 4),
            }
            for d in REFERENCE_DISCOUNT_GRID:
                gt_row[f"true_prob_d{d}"] = round(ref_probs[d], 4)
            gt_rows.append(gt_row)

            # update history AFTER this session's features/outcome are recorded
            if converted:
                previous_purchases += 1
                cumulative_clv += cart_value
                last_purchase_ts = ts
            else:
                previous_abandons += 1
            if assigned_discount > 0:
                discount_event_timestamps.append(ts)
            discount_history_sum += assigned_discount
            discount_history_n += 1

    return pd.DataFrame(tx_rows), pd.DataFrame(gt_rows)


def split_customers_train_test(customer_ids, test_size, seed):
    """Customer-level split (never row-level) so a customer's sessions can
    never appear in both train and test."""
    rng_split = np.random.default_rng(seed)
    ids = np.array(sorted(set(customer_ids)))
    rng_split.shuffle(ids)
    n_test = int(round(len(ids) * test_size))
    test_ids = set(ids[:n_test])
    train_ids = set(ids[n_test:])
    return train_ids, test_ids


def split_continuous_dataset(tx_df, gt_df, test_size, seed):
    train_ids, test_ids = split_customers_train_test(tx_df["customer_id"].unique(), test_size, seed)
    tx_train = tx_df[tx_df["customer_id"].isin(train_ids)].reset_index(drop=True)
    tx_test = tx_df[tx_df["customer_id"].isin(test_ids)].reset_index(drop=True)
    gt_train = gt_df[gt_df["customer_id"].isin(train_ids)].reset_index(drop=True)
    gt_test = gt_df[gt_df["customer_id"].isin(test_ids)].reset_index(drop=True)
    return tx_train, tx_test, gt_train, gt_test, train_ids, test_ids


def validate_continuous_dataset(tx_train, tx_test, gt_train, gt_test, train_ids, test_ids):
    print("=" * 78)
    print("CONTINUOUS DATASET VALIDATION REPORT")
    print("=" * 78)

    tx_all = pd.concat([tx_train, tx_test], ignore_index=True)
    gt_all = pd.concat([gt_train, gt_test], ignore_index=True)

    n_rows = len(tx_all)
    n_customers = tx_all["customer_id"].nunique()
    print(f"Total rows: {n_rows}  (train={len(tx_train)}, test={len(tx_test)})")
    print(f"Total unique customers: {n_customers}  (train={len(train_ids)}, test={len(test_ids)})")

    overlap = train_ids & test_ids
    print(f"Customer overlap between train/test: {len(overlap)}")

    dup_ids = tx_all["transaction_id"].duplicated().sum()
    print(f"Duplicate transaction IDs: {dup_ids}")

    missing = tx_all.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values per column:")
    print(missing if len(missing) else "None")

    d = tx_all["discount_percentage"]
    print(f"\nDiscount percentage: min={d.min():.4f}, max={d.max():.4f}, "
          f"unique values={d.nunique()}, mean={d.mean():.2f}, median={d.median():.2f}")
    print("Discount quantiles:")
    print(d.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nDiscount distribution (text histogram, 20 bins over 0-100%):")
    bins = np.linspace(0, 100, 21)
    hist, edges = np.histogram(d, bins=bins)
    for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(count / max(hist.max(), 1) * 50)
        print(f"  [{lo:5.1f}, {hi:5.1f}) {count:6d} {bar}")

    conv_rate = tx_all["converted"].mean()
    print(f"\nOverall conversion rate: {conv_rate:.3f}")

    prob_cols = [c for c in gt_all.columns if c.startswith("true_prob_d")] + \
        ["true_probability_at_assigned_discount"]
    assert gt_all[prob_cols].min().min() >= 0.0 and gt_all[prob_cols].max().max() <= 1.0, \
        "Ground-truth probabilities out of [0,1] range"

    grid_means = [gt_all[f"true_prob_d{g}"].mean() for g in REFERENCE_DISCOUNT_GRID]
    print("\nAverage true response curve across ALL customers (reference grid):")
    for g, m in zip(REFERENCE_DISCOUNT_GRID, grid_means):
        print(f"  {g:>3}% -> {m:.4f}")
    deltas = [grid_means[i + 1] - grid_means[i] for i in range(len(grid_means) - 1)]
    generally_increasing = sum(x >= -0.005 for x in deltas) >= len(deltas) - 1
    print(f"Generally increasing (allowing one minor noise-driven dip): {generally_increasing}  "
          f"(deltas: {[round(x, 4) for x in deltas]})")
    gain_0_to_20 = grid_means[3] - grid_means[0]
    gain_70_to_100 = grid_means[-1] - grid_means[-4]
    print(f"Gain from 0%->20%: {gain_0_to_20:.4f}   Gain from 70%->100%: {gain_70_to_100:.4f}   "
          f"Diminishing returns: {gain_0_to_20 > gain_70_to_100}")

    sample = tx_all.sample(min(2000, len(tx_all)), random_state=RANDOM_SEED)
    revenue_at_100 = sample["cart_value"] * (1 - 100 / 100.0)
    price_grid = sample["cart_value"].values.reshape(-1, 1) * \
        (1 - np.array(REFERENCE_DISCOUNT_GRID) / 100.0)
    price_never_negative = price_grid.min() >= 0
    print(f"\nRevenue at 100% discount is exactly 0 for all sampled rows: {(revenue_at_100 == 0).all()}")
    print(f"Discounted price never negative across the reference grid: {price_never_negative}")
    cost = sample["cart_value"] * (1 - sample["margin_percentage"] / 100.0)
    profit_at_100 = 0.0 - cost
    print(f"NOTE: under the project's existing cost-based profit formula (profit_if_converted = "
          f"price_after_discount - cost), profit at 100% discount is NOT zero - it equals -cost "
          f"(mean here: {profit_at_100.mean():.2f}); giving the product away loses the merchant its "
          f"full production cost. REVENUE is what hits exactly zero. Flagging this explicitly since "
          f"the acceptance criteria say 'profit ... should therefore be 0' - see the assumptions "
          f"note in the final summary.")

    assert len(overlap) == 0, "Customer overlap between train/test must be zero"
    assert dup_ids == 0, "Duplicate transaction_id values found"
    assert d.min() >= 0.0 and d.max() <= 100.0, "discount_percentage out of [0,100] range"
    assert d.nunique() > 1000, "Expected thousands of unique discount values"
    assert set(tx_all["converted"].unique()) <= {0, 1}, "converted must be binary"
    assert conv_rate < 0.90, "Conversion rate suspiciously high"
    assert (revenue_at_100 == 0).all(), "Revenue at 100% discount must be exactly 0"
    assert price_never_negative, "Discounted price must never go negative"

    print("\nAll continuous-dataset validation checks passed.")
    print("=" * 78)

    return {
        "n_rows": n_rows, "n_customers": n_customers,
        "n_train_rows": len(tx_train), "n_test_rows": len(tx_test),
        "n_train_customers": len(train_ids), "n_test_customers": len(test_ids),
        "customer_overlap": len(overlap),
        "unique_discount_values": int(d.nunique()),
        "discount_min": float(d.min()), "discount_max": float(d.max()),
        "conversion_rate": float(conv_rate),
        "response_curve_means": dict(zip(REFERENCE_DISCOUNT_GRID, [round(m, 4) for m in grid_means])),
    }


def response_curve_sanity_check(customers, gt_df, n_per_segment=2):
    """Section 10: pick representative customers per latent segment and print
    their hidden ground-truth response curve across the reference grid."""
    print("\n" + "=" * 78)
    print("RESPONSE-CURVE SANITY CHECK (representative customers per segment)")
    print("=" * 78)

    rows_out = []
    header = ("Segment          | Customer    | Sensitivity | " +
              " | ".join(f"{g:>4}%" for g in REFERENCE_DISCOUNT_GRID))
    print(header)
    print("-" * len(header))

    for seg in SEGMENTS:
        seg_customers = customers[customers["segment"] == seg].sort_values("price_sensitivity")
        if len(seg_customers) == 0:
            continue
        pick_idx = np.linspace(0, len(seg_customers) - 1, n_per_segment).astype(int)
        picks = seg_customers.iloc[pick_idx]
        for _, row in picks.iterrows():
            cust_gt = gt_df[gt_df["customer_id"] == row["customer_id"]]
            if len(cust_gt) == 0:
                continue
            curve = [cust_gt[f"true_prob_d{g}"].mean() for g in REFERENCE_DISCOUNT_GRID]
            line = (f"{seg:<17}| {row['customer_id']:<12}| {row['price_sensitivity']:.3f}       | " +
                    " | ".join(f"{p:.3f}" for p in curve))
            print(line)
            rows_out.append({
                "segment": seg, "customer_id": row["customer_id"],
                "price_sensitivity": round(row["price_sensitivity"], 4),
                **{f"true_prob_d{g}": round(p, 4) for g, p in zip(REFERENCE_DISCOUNT_GRID, curve)},
            })

    curve_df = pd.DataFrame(rows_out)

    all_increasing = True
    reaches_100pct = False
    for _, r in curve_df.iterrows():
        vals = [r[f"true_prob_d{g}"] for g in REFERENCE_DISCOUNT_GRID]
        diffs = np.diff(vals)
        if (diffs < -0.01).sum() > 1:
            all_increasing = False
        if vals[-1] >= 0.999:
            reaches_100pct = True

    print(f"\nAll sampled curves generally increasing with discount: {all_increasing}")
    print(f"No sampled curve reaches ~100% conversion at 100% discount: {not reaches_100pct}")

    return curve_df


# --------------------------------------------------------------------------------------
# MAIN (continuous - NEW)
# --------------------------------------------------------------------------------------
def main_continuous():
    global rng
    rng = np.random.default_rng(RANDOM_SEED)  # independent of whatever main() may have consumed

    os.makedirs(CONTINUOUS_OUT_DIR, exist_ok=True)

    print("Generating CONTINUOUS-TREATMENT dataset (0-100% discount range)...")
    customers = generate_customers(CONTINUOUS_N_CUSTOMERS)
    session_counts = assign_session_counts(CONTINUOUS_N_CUSTOMERS, CONTINUOUS_N_TRANSACTIONS)

    tx_df, gt_df = generate_continuous_checkout_sessions(customers, session_counts)
    tx_df, gt_df = finalize_and_sort(tx_df, gt_df)

    tx_train, tx_test, gt_train, gt_test, train_ids, test_ids = split_continuous_dataset(
        tx_df, gt_df, CONTINUOUS_TEST_SIZE, RANDOM_SEED
    )

    stats = validate_continuous_dataset(tx_train, tx_test, gt_train, gt_test, train_ids, test_ids)
    curve_df = response_curve_sanity_check(customers, gt_df)

    tx_train_path = os.path.join(CONTINUOUS_OUT_DIR, "transactions_train.csv")
    tx_test_path = os.path.join(CONTINUOUS_OUT_DIR, "transactions_test.csv")
    gt_train_path = os.path.join(CONTINUOUS_OUT_DIR, "ground_truth_train.csv")
    gt_test_path = os.path.join(CONTINUOUS_OUT_DIR, "ground_truth_test.csv")
    curve_path = os.path.join(CONTINUOUS_OUT_DIR, "response_curve_sanity_check.csv")
    metadata_path = os.path.join(CONTINUOUS_OUT_DIR, "metadata.json")

    tx_train.to_csv(tx_train_path, index=False)
    tx_test.to_csv(tx_test_path, index=False)
    gt_train.to_csv(gt_train_path, index=False)
    gt_test.to_csv(gt_test_path, index=False)
    curve_df.to_csv(curve_path, index=False)

    feature_cols = [
        "customer_type", "category", "device_type",
        "previous_purchases", "previous_abandons", "days_since_last_purchase",
        "customer_lifetime_value", "historical_discount_rate",
        "recent_discount_count", "days_since_last_discount",
        "cart_value", "items_count", "margin_percentage",
        "time_on_checkout_seconds", "pages_viewed", "payment_attempts",
        "hour", "day_of_week", "discount_percentage",
    ]
    metadata = {
        "random_seed": RANDOM_SEED,
        "n_customers": CONTINUOUS_N_CUSTOMERS,
        "n_transactions": CONTINUOUS_N_TRANSACTIONS,
        "test_size": CONTINUOUS_TEST_SIZE,
        "feature_cols": feature_cols,
        "categorical_features": ["customer_type", "category", "device_type"],
        "target_col": "converted",
        "treatment_col": "discount_percentage",
        "treatment_range": [0.0, 100.0],
        "reference_discount_grid": REFERENCE_DISCOUNT_GRID,
        "common_round_discounts": COMMON_ROUND_DISCOUNTS,
        "round_discount_prob": ROUND_DISCOUNT_PROB,
        "uniform_mix_prob": UNIFORM_MIX_PROB,
        "beta_skewed_prob": round(1.0 - ROUND_DISCOUNT_PROB - UNIFORM_MIX_PROB, 4),
        "discount_cooldown_days": DISCOUNT_COOLDOWN_DAYS,
        "stats": stats,
        "note": (
            "Ground-truth files store base_logit_no_discount, price_sensitivity, and "
            "previous_abandons_at_eval per row - together these fully determine "
            "true_conversion_probability(base_logit_no_discount, price_sensitivity, "
            "previous_abandons, discount_pct) in src/data_generation.py for ANY discount "
            "in [0,100], not just the reference grid also stored here for convenience."
        ),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\nSaved:")
    print(f"  {tx_train_path}  ({len(tx_train)} rows, {len(train_ids)} customers)")
    print(f"  {tx_test_path}  ({len(tx_test)} rows, {len(test_ids)} customers)")
    print(f"  {gt_train_path}  ({len(gt_train)} rows)")
    print(f"  {gt_test_path}  ({len(gt_test)} rows)")
    print(f"  {curve_path}")
    print(f"  {metadata_path}")

    return stats, curve_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Pricing Agent synthetic data generator")
    parser.add_argument(
        "--mode", choices=["discrete", "continuous", "both"], default="discrete",
        help=("'discrete' (default): reproduces the existing 0/5/10/15%% dataset exactly as "
              "before - running with no arguments is unchanged. 'continuous': generates the "
              "new 0-100%% continuous-discount dataset under data/raw/continuous/, additive "
              "and never overwriting the discrete files. 'both': runs both."),
    )
    args = parser.parse_args()

    if args.mode in ("discrete", "both"):
        main()
    if args.mode in ("continuous", "both"):
        main_continuous()