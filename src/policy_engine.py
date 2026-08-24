"""
src/policy_engine.py

Deterministic, auditable, configurable policy engine for the Dynamic Pricing
Agent. Sits between the T-learner uplift model (src/uplift_model.py) and the
eventual LLM agent / Razorpay integration.

    checkout features -> T-learner -> pred_p0/5/10/15, pred_expected_profit_*,
    pred_optimal_discount  --(THIS FILE)-->  policy decision  --later-->  LLM agent

This module trains nothing and calls no model, no LLM, and no external API.
It takes ML predictions the caller already computed (e.g. by loading the
T-learner's joblib models and scoring a live checkout) and applies business
rules from config/policy.yaml to decide what discount, if any, actually gets
offered.

Profit formula reused verbatim from src/eda.py and src/uplift_model.py:
    cost                 = cart_value * (1 - margin_percentage / 100)
    price_after_discount = cart_value * (1 - discount / 100)
    profit_if_converted  = price_after_discount - cost
    expected_profit(d)   = P(convert | d) * profit_if_converted(d)

Ground truth is never used here — only the ML model's own predictions,
exactly as the caller supplies them in a CheckoutDecisionRequest.

Public entry point:
    from src.policy_engine import evaluate_policy
    decision = evaluate_policy(request)   # request: CheckoutDecisionRequest or dict
"""

import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Union

import yaml

# --------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------
DISCOUNT_LEVELS = [0, 5, 10, 15]
DEFAULT_CONFIG_PATH = "config/policy.yaml"

# Informational only (does not affect policy logic) — identifies which model
# architecture the predictions in an audit record came from.
MODEL_VERSION = "t_learner_gbc_v1"

REASON_CODES = {
    "MODEL_UNAVAILABLE",
    "NO_DISCOUNT_OPTIMAL",
    "INSUFFICIENT_UPLIFT",
    "PROFIT_FLOOR_VIOLATION",
    "DISCOUNT_FREQUENCY_LIMIT",
    "DISCOUNT_CAP_EXCEEDED",
    "HIGH_DISCOUNT_REQUIRES_APPROVAL",
    "APPROVED",
}


# --------------------------------------------------------------------------------------
# DATA MODELS
# --------------------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    max_discount_percent: int = 15
    minimum_profit_margin_percent: float = 5.0
    minimum_uplift: float = 0.03
    max_discounts_per_customer: int = 2
    discount_cooldown_days: int = 7
    human_approval_threshold_percent: int = 10
    fallback_discount_percent: int = 0
    version: str = "1.0.0"


@dataclass
class CheckoutDecisionRequest:
    # --- customer information ---
    customer_id: str
    customer_type: str
    previous_purchases: int
    previous_abandons: int
    historical_discount_rate: float
    recent_discount_count: int
    days_since_last_discount: Optional[float]

    # --- checkout information ---
    cart_value: float
    category: str
    margin_percentage: float
    device_type: str
    time_on_checkout_seconds: float
    pages_viewed: int
    payment_attempts: int
    hour: int
    day_of_week: int

    # --- ML outputs (may be missing/invalid -> triggers RULE 1 fallback) ---
    pred_p0: Optional[float] = None
    pred_p5: Optional[float] = None
    pred_p10: Optional[float] = None
    pred_p15: Optional[float] = None
    pred_expected_profit_0: Optional[float] = None
    pred_expected_profit_5: Optional[float] = None
    pred_expected_profit_10: Optional[float] = None
    pred_expected_profit_15: Optional[float] = None
    pred_optimal_discount: Optional[int] = None
    pred_max_expected_profit: Optional[float] = None


def _coerce_request(request: Union[CheckoutDecisionRequest, dict]) -> CheckoutDecisionRequest:
    if isinstance(request, CheckoutDecisionRequest):
        return request
    if isinstance(request, dict):
        return CheckoutDecisionRequest(**request)
    raise TypeError("request must be a CheckoutDecisionRequest or a dict of its fields")


def _coerce_config(config: Union[PolicyConfig, dict, None]) -> PolicyConfig:
    if config is None:
        return load_policy_config()
    if isinstance(config, PolicyConfig):
        return config
    if isinstance(config, dict):
        return PolicyConfig(**config)
    raise TypeError("config must be a PolicyConfig, a dict, or None")


def load_policy_config(path: str = DEFAULT_CONFIG_PATH) -> PolicyConfig:
    """Load thresholds from policy.yaml. Never hardcode these values elsewhere."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {path}. Run from the project root (the directory containing "
            f"'config/' and 'src/'), or pass an explicit config to evaluate_policy()."
        )
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return PolicyConfig(
        max_discount_percent=raw.get("max_discount_percent", 15),
        minimum_profit_margin_percent=raw.get("minimum_profit_margin_percent", 5.0),
        minimum_uplift=raw.get("minimum_uplift", 0.03),
        max_discounts_per_customer=raw.get("max_discounts_per_customer", 2),
        discount_cooldown_days=raw.get("discount_cooldown_days", 7),
        human_approval_threshold_percent=raw.get("human_approval_threshold_percent", 10),
        fallback_discount_percent=raw.get("fallback_discount_percent", 0),
        version=str(raw.get("policy_version", "1.0.0")),
    )


# --------------------------------------------------------------------------------------
# RULE 1 — MODEL AVAILABILITY
# --------------------------------------------------------------------------------------
def validate_model_output(request: CheckoutDecisionRequest):
    """True/False + a short detail string. Checks presence, finiteness, and
    basic range sanity of every ML field the policy engine depends on."""
    required = {
        "pred_p0": request.pred_p0, "pred_p5": request.pred_p5,
        "pred_p10": request.pred_p10, "pred_p15": request.pred_p15,
        "pred_expected_profit_0": request.pred_expected_profit_0,
        "pred_expected_profit_5": request.pred_expected_profit_5,
        "pred_expected_profit_10": request.pred_expected_profit_10,
        "pred_expected_profit_15": request.pred_expected_profit_15,
        "pred_optimal_discount": request.pred_optimal_discount,
        "pred_max_expected_profit": request.pred_max_expected_profit,
    }
    for name, value in required.items():
        if value is None:
            return False, f"required model output '{name}' is missing (None)"
        if isinstance(value, float) and not math.isfinite(value):
            return False, f"required model output '{name}' is NaN or infinite"

    for name in ("pred_p0", "pred_p5", "pred_p10", "pred_p15"):
        p = required[name]
        if not (0.0 <= p <= 1.0):
            return False, f"'{name}' = {p} is outside the valid [0, 1] probability range"

    if request.pred_optimal_discount not in DISCOUNT_LEVELS:
        return False, (f"pred_optimal_discount = {request.pred_optimal_discount} is not a "
                        f"recognized discount level {DISCOUNT_LEVELS}")

    return True, "model outputs present and valid"


# --------------------------------------------------------------------------------------
# UPLIFT
# --------------------------------------------------------------------------------------
def calculate_uplift(request: CheckoutDecisionRequest) -> dict:
    """uplift(d) = predicted P(convert | d) - predicted P(convert | 0), for d in {5,10,15}.
    0% is exempt from any uplift requirement by definition."""
    return {
        5: request.pred_p5 - request.pred_p0,
        10: request.pred_p10 - request.pred_p0,
        15: request.pred_p15 - request.pred_p0,
    }


# --------------------------------------------------------------------------------------
# RULE 3 — PROFIT FLOOR
# --------------------------------------------------------------------------------------
def check_profit_floor(request: CheckoutDecisionRequest, config: PolicyConfig, discount: int):
    """Same economic definition as src/eda.py and src/uplift_model.py:
    cost = V*(1-m), price_after_discount = V*(1-d/100), profit_if_converted =
    price_after_discount - cost. Passes if profit_if_converted / cart_value
    is still at or above the configured floor."""
    V = request.cart_value
    m = request.margin_percentage / 100.0
    cost = V * (1 - m)
    price_after_discount = V * (1 - discount / 100.0)
    profit_if_converted = price_after_discount - cost

    margin_after_discount_pct = (profit_if_converted / V) * 100 if V else float("-inf")
    passed = margin_after_discount_pct >= config.minimum_profit_margin_percent
    detail = (f"margin after {discount}% discount = {margin_after_discount_pct:.2f}% "
              f"(floor = {config.minimum_profit_margin_percent:.2f}%)")
    return passed, detail


# --------------------------------------------------------------------------------------
# RULE 6 — CUSTOMER DISCOUNT FREQUENCY
# --------------------------------------------------------------------------------------
def check_frequency_limit(request: CheckoutDecisionRequest, config: PolicyConfig):
    """Only uses recent_discount_count, which the caller is expected to have
    already scoped to the configured cooldown window — this function does not
    invent cooldown logic from days_since_last_discount, per spec."""
    blocked = request.recent_discount_count >= config.max_discounts_per_customer
    if blocked:
        detail = (f"customer has received {request.recent_discount_count} discount(s), "
                  f"at or above the limit of {config.max_discounts_per_customer} within "
                  f"the configured {config.discount_cooldown_days}-day cooldown")
    else:
        detail = (f"customer has received {request.recent_discount_count} discount(s), "
                  f"under the limit of {config.max_discounts_per_customer}")
    return (not blocked), detail


# --------------------------------------------------------------------------------------
# CANDIDATE FILTERING + ECONOMIC SELECTION
# (RULE 2 discount bound, RULE 3 profit floor, RULE 4 uplift, RULE 6 frequency)
# --------------------------------------------------------------------------------------
def select_best_valid_discount(request: CheckoutDecisionRequest, config: PolicyConfig):
    """Considers ALL FOUR candidate discounts, filters each against every
    applicable policy rule, and returns the highest-predicted-expected-profit
    candidate that survives. 0% is always structurally valid (never subject to
    the cap, profit-floor, or uplift checks; only a customer-level frequency
    block is even possible for it, and 0% is never itself a 'discount received').
    """
    profit_map = {
        0: request.pred_expected_profit_0, 5: request.pred_expected_profit_5,
        10: request.pred_expected_profit_10, 15: request.pred_expected_profit_15,
    }
    uplift = calculate_uplift(request)

    freq_ok, freq_detail = check_frequency_limit(request, config)
    checks = [{"check": "frequency_limit", "discount": None, "passed": freq_ok, "detail": freq_detail}]

    valid = {0: True}
    exclusion_reasons = {}

    for d in (5, 10, 15):
        reasons = []

        cap_ok = d <= config.max_discount_percent
        checks.append({
            "check": "discount_cap", "discount": d, "passed": cap_ok,
            "detail": (f"{d}% is within the configured max of {config.max_discount_percent}%" if cap_ok
                       else f"{d}% exceeds the configured max of {config.max_discount_percent}%"),
        })
        if not cap_ok:
            reasons.append("DISCOUNT_CAP_EXCEEDED")

        floor_ok, floor_detail = check_profit_floor(request, config, d)
        checks.append({"check": "profit_floor", "discount": d, "passed": floor_ok, "detail": floor_detail})
        if not floor_ok:
            reasons.append("PROFIT_FLOOR_VIOLATION")

        d_uplift = uplift[d]
        uplift_ok = d_uplift >= config.minimum_uplift
        checks.append({
            "check": "minimum_uplift", "discount": d, "passed": uplift_ok,
            "detail": (f"uplift {d_uplift:.4f} >= required {config.minimum_uplift:.4f}" if uplift_ok
                       else f"uplift {d_uplift:.4f} < required {config.minimum_uplift:.4f}"),
        })
        if not uplift_ok:
            reasons.append("INSUFFICIENT_UPLIFT")

        if not freq_ok:
            reasons.append("DISCOUNT_FREQUENCY_LIMIT")

        valid[d] = len(reasons) == 0
        if reasons:
            exclusion_reasons[d] = reasons

    valid_candidates = [d for d in DISCOUNT_LEVELS if valid[d]]
    best = max(valid_candidates, key=lambda d: profit_map[d])

    if best == 0:
        if len(valid_candidates) > 1:
            # 0% legitimately out-profited at least one other still-valid candidate
            provisional_reason = "NO_DISCOUNT_OPTIMAL"
        else:
            # every positive discount was excluded outright - report the most
            # decisive binding reason (a hard customer-level rule outranks the
            # economic ones, which in turn outrank the structural cap)
            all_reasons = set()
            for d in (5, 10, 15):
                all_reasons.update(exclusion_reasons.get(d, []))
            for candidate_reason in ("DISCOUNT_FREQUENCY_LIMIT", "PROFIT_FLOOR_VIOLATION",
                                      "INSUFFICIENT_UPLIFT", "DISCOUNT_CAP_EXCEEDED"):
                if candidate_reason in all_reasons:
                    provisional_reason = candidate_reason
                    break
            else:
                provisional_reason = "NO_DISCOUNT_OPTIMAL"
    else:
        provisional_reason = "APPROVED"

    return best, provisional_reason, checks, request.pred_optimal_discount, profit_map, uplift


# --------------------------------------------------------------------------------------
# RULE 7 / RULE 8 — HUMAN APPROVAL GATE + FINAL APPROVAL
# --------------------------------------------------------------------------------------
def apply_approval_gate(candidate_discount: int, provisional_reason_code: str, config: PolicyConfig):
    if candidate_discount > config.human_approval_threshold_percent:
        return "HUMAN_APPROVAL_REQUIRED", "HIGH_DISCOUNT_REQUIRES_APPROVAL"
    return "APPROVED", provisional_reason_code


# --------------------------------------------------------------------------------------
# DECISION EXPLANATION
# --------------------------------------------------------------------------------------
def _generate_explanation(reason_code: str, discount: int) -> str:
    templates = {
        "MODEL_UNAVAILABLE":
            "Safe fallback applied because the pricing model was unavailable.",
        "NO_DISCOUNT_OPTIMAL":
            "No discount approved because the predicted incremental conversion lift "
            "does not justify reducing the merchant's margin.",
        "INSUFFICIENT_UPLIFT":
            f"Discount rejected because the predicted incremental conversion lift does not "
            f"meet the minimum required threshold; {discount}% is the best option that qualifies.",
        "PROFIT_FLOOR_VIOLATION":
            f"Discount rejected because it would push the merchant's margin below the "
            f"configured profit floor; {discount}% is the highest discount that still clears it.",
        "DISCOUNT_FREQUENCY_LIMIT":
            "Discount rejected because this customer has reached the configured discount "
            "frequency limit.",
        "DISCOUNT_CAP_EXCEEDED":
            f"Discount capped at {discount}% because higher discounts exceed the maximum "
            f"allowed by policy.",
        "HIGH_DISCOUNT_REQUIRES_APPROVAL":
            f"{discount}% discount requires human approval because it exceeds the "
            f"automatic approval threshold.",
        "APPROVED":
            f"{discount}% discount approved because it provides sufficient incremental "
            f"conversion lift while remaining above the merchant's profit floor.",
    }
    return templates.get(reason_code, f"Decision reason: {reason_code}; discount: {discount}%.")


# --------------------------------------------------------------------------------------
# AUDIT RECORD
# --------------------------------------------------------------------------------------
def build_audit_record(request: CheckoutDecisionRequest, config: PolicyConfig, *, decision: str,
                        reason_code: str, policy_selected_discount: int, checks: list,
                        model_recommended_discount, model_available: bool,
                        profit_map: Optional[dict] = None, uplift: Optional[dict] = None) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()

    if not model_available:
        pred_probs = {f"pred_p{d}": None for d in DISCOUNT_LEVELS}
        pred_profits = {f"pred_expected_profit_{d}": None for d in DISCOUNT_LEVELS}
        expected_profit_selected = None
        predicted_uplift_selected = None
    else:
        pred_probs = {
            "pred_p0": request.pred_p0, "pred_p5": request.pred_p5,
            "pred_p10": request.pred_p10, "pred_p15": request.pred_p15,
        }
        pred_profits = {
            "pred_expected_profit_0": request.pred_expected_profit_0,
            "pred_expected_profit_5": request.pred_expected_profit_5,
            "pred_expected_profit_10": request.pred_expected_profit_10,
            "pred_expected_profit_15": request.pred_expected_profit_15,
        }
        expected_profit_selected = profit_map[policy_selected_discount]
        predicted_uplift_selected = 0.0 if policy_selected_discount == 0 else uplift[policy_selected_discount]

    record = {
        "timestamp": timestamp,
        "customer_id": request.customer_id,
        "cart_value": request.cart_value,
        **pred_probs,
        **pred_profits,
        "model_recommended_discount": model_recommended_discount,
        "policy_selected_discount": policy_selected_discount,
        "decision": decision,
        "reason_code": reason_code,
        "policy_checks": checks,
        "expected_profit_selected": expected_profit_selected,
        "predicted_uplift_selected": predicted_uplift_selected,
        "model_version": MODEL_VERSION,
        "policy_version": config.version,
        "model_available": model_available,
        "explanation": _generate_explanation(reason_code, policy_selected_discount),
    }
    return record


# --------------------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# --------------------------------------------------------------------------------------
def evaluate_policy(request: Union[CheckoutDecisionRequest, dict],
                     config: Union[PolicyConfig, dict, None] = None) -> dict:
    """Evaluate one checkout's pricing decision. Deterministic, side-effect-free
    (no model calls, no LLM calls, no network calls), and never raises for
    missing/invalid ML predictions -- RULE 1 catches those and returns a safe
    fallback instead, so a checkout can never fail because of this function."""
    request = _coerce_request(request)
    config = _coerce_config(config)

    # RULE 1 - model availability (short-circuits everything else)
    valid_model, model_detail = validate_model_output(request)
    if not valid_model:
        checks = [{"check": "model_availability", "discount": None, "passed": False, "detail": model_detail}]
        return build_audit_record(
            request, config,
            decision="FALLBACK", reason_code="MODEL_UNAVAILABLE",
            policy_selected_discount=config.fallback_discount_percent,
            checks=checks, model_recommended_discount=None, model_available=False,
        )

    # RULES 2/3/4/5/6 - filter all four candidates, pick the best valid one
    best, provisional_reason, checks, model_recommended, profit_map, uplift = \
        select_best_valid_discount(request, config)
    checks = [{"check": "model_availability", "discount": None, "passed": True, "detail": model_detail}] + checks

    # RULES 7/8 - human approval gate / final approval
    decision, reason_code = apply_approval_gate(best, provisional_reason, config)

    return build_audit_record(
        request, config,
        decision=decision, reason_code=reason_code, policy_selected_discount=best,
        checks=checks, model_recommended_discount=model_recommended,
        model_available=True, profit_map=profit_map, uplift=uplift,
    )