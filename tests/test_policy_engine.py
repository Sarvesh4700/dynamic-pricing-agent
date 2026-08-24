"""
tests/test_policy_engine.py

Covers the 12 required scenarios for the Dynamic Pricing Agent policy engine.
Run from the project root:
    python -m pytest tests/test_policy_engine.py -v
"""

import math

import pytest

from src.policy_engine import PolicyConfig, evaluate_policy


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def _expected_profit(cart_value, margin_percentage, discount, prob):
    """Mirrors the exact formula in src/eda.py, src/uplift_model.py, and
    src/policy_engine.py, so test fixtures stay internally consistent."""
    V = cart_value
    m = margin_percentage / 100.0
    cost = V * (1 - m)
    price_after_discount = V * (1 - discount / 100.0)
    profit_if_converted = price_after_discount - cost
    return prob * profit_if_converted


def make_request(probs, cart_value=1000.0, margin_percentage=30.0, recent_discount_count=0,
                  **overrides):
    """Builds a full CheckoutDecisionRequest-shaped dict from just the four
    potential-outcome probabilities plus cart/margin, auto-deriving the
    expected-profit fields and pred_optimal_discount so numbers can never
    drift out of sync with the formula the engine itself uses."""
    profits = {d: _expected_profit(cart_value, margin_percentage, d, probs[d]) for d in (0, 5, 10, 15)}
    optimal = max(profits, key=profits.get)

    req = {
        "customer_id": "C_TEST_001",
        "customer_type": "returning",
        "previous_purchases": 3,
        "previous_abandons": 1,
        "historical_discount_rate": 5.0,
        "recent_discount_count": recent_discount_count,
        "days_since_last_discount": 30,
        "cart_value": cart_value,
        "category": "fashion",
        "margin_percentage": margin_percentage,
        "device_type": "desktop",
        "time_on_checkout_seconds": 90.0,
        "pages_viewed": 3,
        "payment_attempts": 1,
        "hour": 14,
        "day_of_week": 2,
        "pred_p0": probs[0], "pred_p5": probs[5], "pred_p10": probs[10], "pred_p15": probs[15],
        "pred_expected_profit_0": profits[0], "pred_expected_profit_5": profits[5],
        "pred_expected_profit_10": profits[10], "pred_expected_profit_15": profits[15],
        "pred_optimal_discount": optimal,
        "pred_max_expected_profit": profits[optimal],
    }
    req.update(overrides)
    return req


DEFAULT_CONFIG = PolicyConfig()  # mirrors config/policy.yaml defaults


# --------------------------------------------------------------------------------------
# 1. 0% optimal -> approve 0%
# --------------------------------------------------------------------------------------
def test_zero_percent_optimal_is_approved():
    # 0% clearly has the best predicted profit even though 5/10/15 are all
    # individually valid (sufficient uplift, healthy margin) - so 0% wins on
    # economics alone, not because anything else was excluded.
    probs = {0: 0.80, 5: 0.85, 10: 0.86, 15: 0.87}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0)

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["decision"] == "APPROVED"
    assert result["policy_selected_discount"] == 0
    assert result["reason_code"] == "NO_DISCOUNT_OPTIMAL"


# --------------------------------------------------------------------------------------
# 2. 5% profitable and sufficient uplift -> approve 5%
# --------------------------------------------------------------------------------------
def test_five_percent_approved_when_most_profitable():
    probs = {0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0)

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["decision"] == "APPROVED"
    assert result["policy_selected_discount"] == 5
    assert result["reason_code"] == "APPROVED"


# --------------------------------------------------------------------------------------
# 3. Discount with insufficient uplift -> reject that discount
# --------------------------------------------------------------------------------------
def test_insufficient_uplift_rejects_that_discount():
    # 15% barely moves conversion at all (uplift 0.01 < the 0.03 minimum) so it
    # must be excluded from consideration, even though 5%/10% remain valid.
    probs = {0: 0.30, 5: 0.45, 10: 0.47, 15: 0.31}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0)

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["policy_selected_discount"] != 15
    uplift_checks = [c for c in result["policy_checks"]
                      if c["check"] == "minimum_uplift" and c["discount"] == 15]
    assert len(uplift_checks) == 1
    assert uplift_checks[0]["passed"] is False


# --------------------------------------------------------------------------------------
# 4. Discount violating profit floor -> reject it, select best valid lower discount
# --------------------------------------------------------------------------------------
def test_profit_floor_violation_falls_back_to_best_valid_lower_discount():
    # 12% margin: 5% discount still clears the 5% floor (12-5=7 >= 5), but
    # 10% and 15% do not (12-10=2 and 12-15=-3, both < 5).
    probs = {0: 0.15, 5: 0.45, 10: 0.55, 15: 0.60}
    req = make_request(probs, cart_value=1000.0, margin_percentage=12.0)

    result = evaluate_policy(req, DEFAULT_CONFIG)

    floor_checks = {c["discount"]: c["passed"] for c in result["policy_checks"]
                    if c["check"] == "profit_floor" and c["discount"] in (10, 15)}
    assert floor_checks[10] is False
    assert floor_checks[15] is False
    assert result["policy_selected_discount"] == 5
    assert result["decision"] == "APPROVED"


# --------------------------------------------------------------------------------------
# 5. 15% recommendation -> human approval required
# --------------------------------------------------------------------------------------
def test_high_discount_requires_human_approval():
    probs = {0: 0.20, 5: 0.35, 10: 0.55, 15: 0.75}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0)

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["policy_selected_discount"] == 15
    assert result["decision"] == "HUMAN_APPROVAL_REQUIRED"
    assert result["reason_code"] == "HIGH_DISCOUNT_REQUIRES_APPROVAL"


# --------------------------------------------------------------------------------------
# 6. Customer at discount frequency limit -> reject discount
# --------------------------------------------------------------------------------------
def test_frequency_limit_blocks_new_discount():
    # Same probabilities as test 2 (would otherwise clearly pick 5%), but this
    # customer has already hit the configured max_discounts_per_customer (2).
    probs = {0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0, recent_discount_count=2)

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["policy_selected_discount"] == 0
    assert result["decision"] == "APPROVED"
    assert result["reason_code"] == "DISCOUNT_FREQUENCY_LIMIT"


# --------------------------------------------------------------------------------------
# 7. Model unavailable -> safe 0% fallback
# --------------------------------------------------------------------------------------
def test_model_unavailable_triggers_safe_fallback():
    probs = {0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0)
    req["pred_p0"] = None  # simulate a missing model output

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["decision"] == "FALLBACK"
    assert result["policy_selected_discount"] == 0
    assert result["reason_code"] == "MODEL_UNAVAILABLE"
    assert result["model_available"] is False


# --------------------------------------------------------------------------------------
# 8. Discount above maximum allowed -> never approve above the configured cap
# --------------------------------------------------------------------------------------
def test_never_approves_above_configured_max_discount():
    # Same numbers as test 5, where 15% would otherwise be the clear economic
    # winner, but this merchant's policy caps discounts at 10%.
    probs = {0: 0.20, 5: 0.35, 10: 0.55, 15: 0.75}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0)
    capped_config = PolicyConfig(max_discount_percent=10)

    result = evaluate_policy(req, capped_config)

    assert result["policy_selected_discount"] <= 10
    assert result["policy_selected_discount"] != 15
    cap_check_15 = [c for c in result["policy_checks"]
                     if c["check"] == "discount_cap" and c["discount"] == 15][0]
    assert cap_check_15["passed"] is False


# --------------------------------------------------------------------------------------
# 9. Multiple valid discounts -> choose the one with highest predicted expected profit
# --------------------------------------------------------------------------------------
def test_selects_highest_expected_profit_among_valid_candidates():
    # 0/5/10/15 all individually valid (sufficient uplift, healthy margin);
    # 10% has the highest predicted expected profit of the four.
    probs = {0: 0.30, 5: 0.40, 10: 0.58, 15: 0.60}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0)

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["policy_selected_discount"] == 10
    assert result["decision"] == "APPROVED"


# --------------------------------------------------------------------------------------
# 10. All positive discounts invalid -> return 0%
# --------------------------------------------------------------------------------------
def test_all_positive_discounts_invalid_returns_zero():
    # Healthy margin (so profit floor is never the issue), but every positive
    # discount lifts conversion by less than the 0.03 minimum uplift.
    probs = {0: 0.50, 5: 0.51, 10: 0.515, 15: 0.52}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0)

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["policy_selected_discount"] == 0
    assert result["reason_code"] == "INSUFFICIENT_UPLIFT"


# --------------------------------------------------------------------------------------
# 11. Missing/NaN model probability -> fallback safely
# --------------------------------------------------------------------------------------
def test_nan_probability_triggers_safe_fallback():
    probs = {0: 0.30, 5: 0.42, 10: float("nan"), 15: 0.50}
    req = make_request({0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50}, cart_value=1000.0, margin_percentage=30.0)
    req["pred_p10"] = float("nan")

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["decision"] == "FALLBACK"
    assert result["policy_selected_discount"] == 0
    assert result["reason_code"] == "MODEL_UNAVAILABLE"


# --------------------------------------------------------------------------------------
# 12. Audit record contains required fields
# --------------------------------------------------------------------------------------
def test_audit_record_contains_required_fields():
    probs = {0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50}
    req = make_request(probs, cart_value=1000.0, margin_percentage=30.0)

    result = evaluate_policy(req, DEFAULT_CONFIG)

    required_fields = [
        "timestamp", "customer_id", "cart_value",
        "pred_p0", "pred_p5", "pred_p10", "pred_p15",
        "pred_expected_profit_0", "pred_expected_profit_5",
        "pred_expected_profit_10", "pred_expected_profit_15",
        "model_recommended_discount", "policy_selected_discount",
        "decision", "reason_code", "policy_checks",
        "expected_profit_selected", "predicted_uplift_selected",
        "model_version", "policy_version",
    ]
    for field in required_fields:
        assert field in result, f"missing required audit field: {field}"

    assert isinstance(result["policy_checks"], list)
    assert len(result["policy_checks"]) > 0


# --------------------------------------------------------------------------------------
# EXTRA: fallback audit record is also well-formed (spec: "clearly indicate
# that the model was unavailable")
# --------------------------------------------------------------------------------------
def test_fallback_audit_record_indicates_model_unavailable():
    req = make_request({0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50}, cart_value=1000.0, margin_percentage=30.0)
    req["pred_optimal_discount"] = None

    result = evaluate_policy(req, DEFAULT_CONFIG)

    assert result["model_available"] is False
    assert result["decision"] == "FALLBACK"
    assert "unavailable" in result["explanation"].lower()
    assert all(result[f"pred_p{d}"] is None for d in (0, 5, 10, 15))