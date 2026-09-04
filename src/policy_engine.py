"""
Deterministic, auditable, configurable policy engine for Dynamic Pricing.

Supports BOTH:

1. Legacy T-learner predictions:
       pred_p0 / pred_p5 / pred_p10 / pred_p15

2. Continuous response-model predictions:
       discount_predictions
       expected_profit_predictions

The policy engine itself never calls an ML model.
It only evaluates predictions supplied by the caller.
"""

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union

import yaml


# =============================================================================
# CONSTANTS
# =============================================================================

DISCOUNT_LEVELS = [0, 5, 10, 15]

DEFAULT_CONFIG_PATH = "config/policy.yaml"

MODEL_VERSION = "continuous_response_v1"

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


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PolicyConfig:
    max_discount_percent: float = 15.0
    minimum_profit_margin_percent: float = 5.0
    minimum_uplift: float = 0.03
    max_discounts_per_customer: int = 2
    discount_cooldown_days: int = 7
    human_approval_threshold_percent: float = 10.0
    fallback_discount_percent: float = 0.0
    version: str = "1.0.0"


@dataclass
class CheckoutDecisionRequest:

    # -------------------------------------------------------------------------
    # Customer information
    # -------------------------------------------------------------------------

    customer_id: str
    customer_type: str
    previous_purchases: int
    previous_abandons: int
    historical_discount_rate: float
    recent_discount_count: int
    days_since_last_discount: Optional[float]

    # -------------------------------------------------------------------------
    # Checkout information
    # -------------------------------------------------------------------------

    cart_value: float
    category: str
    margin_percentage: float
    device_type: str
    time_on_checkout_seconds: float
    pages_viewed: int
    payment_attempts: int
    hour: int
    day_of_week: int

    # -------------------------------------------------------------------------
    # LEGACY MODEL OUTPUTS
    # -------------------------------------------------------------------------

    pred_p0: Optional[float] = None
    pred_p5: Optional[float] = None
    pred_p10: Optional[float] = None
    pred_p15: Optional[float] = None

    pred_expected_profit_0: Optional[float] = None
    pred_expected_profit_5: Optional[float] = None
    pred_expected_profit_10: Optional[float] = None
    pred_expected_profit_15: Optional[float] = None

    # -------------------------------------------------------------------------
    # Common model outputs
    # -------------------------------------------------------------------------

    pred_optimal_discount: Optional[float] = None
    pred_max_expected_profit: Optional[float] = None

    # -------------------------------------------------------------------------
    # CONTINUOUS MODEL OUTPUTS
    # -------------------------------------------------------------------------

    discount_predictions: Optional[dict] = None
    expected_profit_predictions: Optional[dict] = None


# =============================================================================
# REQUEST / CONFIG HELPERS
# =============================================================================

def _coerce_request(
    request: Union[CheckoutDecisionRequest, dict]
) -> CheckoutDecisionRequest:

    if isinstance(request, CheckoutDecisionRequest):
        return request

    if isinstance(request, dict):
        return CheckoutDecisionRequest(**request)

    raise TypeError(
        "request must be a CheckoutDecisionRequest or a dict of its fields"
    )


def _coerce_config(
    config: Union[PolicyConfig, dict, None]
) -> PolicyConfig:

    if config is None:
        return load_policy_config()

    if isinstance(config, PolicyConfig):
        return config

    if isinstance(config, dict):
        return PolicyConfig(**config)

    raise TypeError(
        "config must be a PolicyConfig, a dict, or None"
    )


def load_policy_config(
    path: str = DEFAULT_CONFIG_PATH
) -> PolicyConfig:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {path}. Run from the project root "
            "(the directory containing config/ and src/), "
            "or pass an explicit config to evaluate_policy()."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return PolicyConfig(
        max_discount_percent=float(
            raw.get("max_discount_percent", 15)
        ),
        minimum_profit_margin_percent=float(
            raw.get("minimum_profit_margin_percent", 5.0)
        ),
        minimum_uplift=float(
            raw.get("minimum_uplift", 0.03)
        ),
        max_discounts_per_customer=int(
            raw.get("max_discounts_per_customer", 2)
        ),
        discount_cooldown_days=int(
            raw.get("discount_cooldown_days", 7)
        ),
        human_approval_threshold_percent=float(
            raw.get("human_approval_threshold_percent", 10)
        ),
        fallback_discount_percent=float(
            raw.get("fallback_discount_percent", 0)
        ),
        version=str(
            raw.get("policy_version", "1.0.0")
        ),
    )


# =============================================================================
# PREDICTION HELPERS
# =============================================================================

def _normalise_dict(values: dict) -> dict:

    result = {}

    for key, value in values.items():

        try:
            discount = float(key)
            prediction = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid prediction entry: {key} -> {value}"
            )

        if not math.isfinite(discount):
            raise ValueError(
                f"Discount {key} is not finite"
            )

        if not math.isfinite(prediction):
            raise ValueError(
                f"Prediction for {key}% is not finite"
            )

        if not 0.0 <= prediction <= 1.0:
            raise ValueError(
                f"Prediction for {key}% = {prediction} "
                "is outside [0, 1]"
            )

        result[discount] = prediction

    return result


def get_prediction_map(
    request: CheckoutDecisionRequest
) -> dict:
    """
    Prefer continuous predictions when available.

    Otherwise use the legacy 0/5/10/15 predictions.
    """

    if request.discount_predictions:

        return _normalise_dict(
            request.discount_predictions
        )

    legacy = {
        0.0: request.pred_p0,
        5.0: request.pred_p5,
        10.0: request.pred_p10,
        15.0: request.pred_p15,
    }

    if all(
        value is not None
        for value in legacy.values()
    ):
        return _normalise_dict(
            legacy
        )

    return {}


def get_profit_map(
    request: CheckoutDecisionRequest
) -> dict:

    if request.expected_profit_predictions:

        raw = {}

        for key, value in (
            request.expected_profit_predictions.items()
        ):

            try:
                d = float(key)
                p = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid expected-profit entry: "
                    f"{key} -> {value}"
                )

            if not math.isfinite(p):
                raise ValueError(
                    f"Expected profit for {d}% "
                    "is not finite"
                )

            raw[d] = p

        return raw

    legacy = {
        0.0: request.pred_expected_profit_0,
        5.0: request.pred_expected_profit_5,
        10.0: request.pred_expected_profit_10,
        15.0: request.pred_expected_profit_15,
    }

    if all(
        value is not None
        for value in legacy.values()
    ):
        return {
            float(k): float(v)
            for k, v in legacy.items()
        }

    return {}


# =============================================================================
# RULE 1 — MODEL AVAILABILITY
# =============================================================================

def validate_model_output(
    request: CheckoutDecisionRequest
):

    try:

        probabilities = get_prediction_map(
            request
        )

        profits = get_profit_map(
            request
        )

    except (TypeError, ValueError) as exc:

        return False, str(exc)

    if not probabilities:
        return False, (
            "no valid model conversion predictions "
            "were supplied"
        )

    if not profits:
        return False, (
            "no valid model expected-profit predictions "
            "were supplied"
        )

    if 0.0 not in probabilities:
        return False, (
            "model predictions must include 0% baseline"
        )

    if 0.0 not in profits:
        return False, (
            "model expected profits must include 0% baseline"
        )

    if request.pred_optimal_discount is None:
        return False, (
            "pred_optimal_discount is missing"
        )

    if request.pred_max_expected_profit is None:
        return False, (
            "pred_max_expected_profit is missing"
        )

    try:

        optimal = float(
            request.pred_optimal_discount
        )

        max_profit = float(
            request.pred_max_expected_profit
        )

    except (TypeError, ValueError):

        return False, (
            "pred_optimal_discount or "
            "pred_max_expected_profit is invalid"
        )

    if not math.isfinite(optimal):
        return False, (
            "pred_optimal_discount is not finite"
        )

    if not math.isfinite(max_profit):
        return False, (
            "pred_max_expected_profit is not finite"
        )

    return True, (
        "model outputs present and valid"
    )


# =============================================================================
# UPLIFT
# =============================================================================

def calculate_uplift(
    request: CheckoutDecisionRequest
) -> dict:

    probabilities = get_prediction_map(
        request
    )

    if 0.0 not in probabilities:
        raise ValueError(
            "0% prediction is required to calculate uplift"
        )

    baseline = probabilities[0.0]

    return {
        discount:
            probability - baseline
        for discount, probability
        in probabilities.items()
    }


# =============================================================================
# RULE 3 — PROFIT FLOOR
# =============================================================================

def check_profit_floor(
    request: CheckoutDecisionRequest,
    config: PolicyConfig,
    discount: float,
):

    V = float(
        request.cart_value
    )

    margin = (
        float(request.margin_percentage)
        / 100.0
    )

    if V <= 0:
        return False, (
            "cart value must be greater than zero"
        )

    cost = V * (
        1.0 - margin
    )

    price_after_discount = V * (
        1.0 - float(discount) / 100.0
    )

    profit_if_converted = (
        price_after_discount
        - cost
    )

    margin_after_discount_pct = (
        profit_if_converted
        / V
    ) * 100.0

    passed = (
        margin_after_discount_pct
        >= config.minimum_profit_margin_percent
    )

    detail = (
        f"margin after {discount:g}% discount = "
        f"{margin_after_discount_pct:.2f}% "
        f"(floor = "
        f"{config.minimum_profit_margin_percent:.2f}%)"
    )

    return passed, detail


# =============================================================================
# RULE 6 — FREQUENCY
# =============================================================================

def check_frequency_limit(
    request: CheckoutDecisionRequest,
    config: PolicyConfig,
):

    blocked = (
        request.recent_discount_count
        >= config.max_discounts_per_customer
    )

    if blocked:

        detail = (
            f"customer has received "
            f"{request.recent_discount_count} discount(s), "
            f"at or above the limit of "
            f"{config.max_discounts_per_customer} "
            f"within the configured "
            f"{config.discount_cooldown_days}-day cooldown"
        )

    else:

        detail = (
            f"customer has received "
            f"{request.recent_discount_count} discount(s), "
            f"under the limit of "
            f"{config.max_discounts_per_customer}"
        )

    return (
        not blocked,
        detail,
    )


# =============================================================================
# CANDIDATE FILTERING
# =============================================================================

def select_best_valid_discount(
    request: CheckoutDecisionRequest,
    config: PolicyConfig,
):

    probabilities = get_prediction_map(
        request
    )

    profit_map = get_profit_map(
        request
    )

    uplift = calculate_uplift(
        request
    )

    frequency_ok, frequency_detail = (
        check_frequency_limit(
            request,
            config,
        )
    )

    checks = [
        {
            "check": "frequency_limit",
            "discount": None,
            "passed": frequency_ok,
            "detail": frequency_detail,
        }
    ]

    valid_candidates = []

    exclusion_reasons = {}

    # Only evaluate discounts for which the model supplied
    # both probability and expected profit.
    candidates = sorted(
        set(probabilities.keys())
        & set(profit_map.keys())
    )

    for discount in candidates:

        reasons = []

        # ---------------------------------------------------------------------
        # Discount cap
        # ---------------------------------------------------------------------

        cap_ok = (
            discount
            <= config.max_discount_percent
        )

        checks.append(
            {
                "check": "discount_cap",
                "discount": discount,
                "passed": cap_ok,
                "detail": (
                    f"{discount:g}% is within the configured "
                    f"max of "
                    f"{config.max_discount_percent:g}%"
                    if cap_ok
                    else
                    f"{discount:g}% exceeds the configured "
                    f"max of "
                    f"{config.max_discount_percent:g}%"
                ),
            }
        )

        if not cap_ok:
            reasons.append(
                "DISCOUNT_CAP_EXCEEDED"
            )

        # ---------------------------------------------------------------------
        # Profit floor
        # ---------------------------------------------------------------------

        floor_ok, floor_detail = (
            check_profit_floor(
                request,
                config,
                discount,
            )
        )

        checks.append(
            {
                "check": "profit_floor",
                "discount": discount,
                "passed": floor_ok,
                "detail": floor_detail,
            }
        )

        if not floor_ok and discount != 0:
            reasons.append(
                "PROFIT_FLOOR_VIOLATION"
            )

        # ---------------------------------------------------------------------
        # Uplift
        # ---------------------------------------------------------------------

        if discount == 0:

            uplift_ok = True

            uplift_detail = (
                "0% baseline is exempt from "
                "minimum uplift requirement"
            )

        else:

            d_uplift = uplift.get(
                discount
            )

            if d_uplift is None:

                uplift_ok = False

                uplift_detail = (
                    f"no uplift prediction available "
                    f"for {discount:g}%"
                )

            else:

                uplift_ok = (
                    d_uplift
                    >= config.minimum_uplift
                )

                uplift_detail = (
                    f"uplift {d_uplift:.4f} >= "
                    f"required "
                    f"{config.minimum_uplift:.4f}"
                    if uplift_ok
                    else
                    f"uplift {d_uplift:.4f} < "
                    f"required "
                    f"{config.minimum_uplift:.4f}"
                )

            if not uplift_ok:
                reasons.append(
                    "INSUFFICIENT_UPLIFT"
                )

        checks.append(
            {
                "check": "minimum_uplift",
                "discount": discount,
                "passed": uplift_ok,
                "detail": uplift_detail,
            }
        )

        # ---------------------------------------------------------------------
        # Frequency
        # ---------------------------------------------------------------------

        if (
            discount != 0
            and not frequency_ok
        ):

            reasons.append(
                "DISCOUNT_FREQUENCY_LIMIT"
            )

        # ---------------------------------------------------------------------
        # Candidate accepted/rejected
        # ---------------------------------------------------------------------

        if reasons:

            exclusion_reasons[
                discount
            ] = reasons

        else:

            valid_candidates.append(
                discount
            )

    # -------------------------------------------------------------------------
    # 0% safety fallback
    # -------------------------------------------------------------------------

    if (
        0.0 in profit_map
        and 0.0 not in valid_candidates
    ):
        valid_candidates.append(
            0.0
        )

    if not valid_candidates:

        return (
            0.0,
            "NO_DISCOUNT_OPTIMAL",
            checks,
            request.pred_optimal_discount,
            profit_map,
            uplift,
        )

    # -------------------------------------------------------------------------
    # Economic selection
    # -------------------------------------------------------------------------

    best = max(
        valid_candidates,
        key=lambda d: profit_map[d],
    )

    # -------------------------------------------------------------------------
    # Reason
    # -------------------------------------------------------------------------

    if best == 0.0:

        if len(valid_candidates) > 1:

            provisional_reason = (
                "NO_DISCOUNT_OPTIMAL"
            )

        else:

            all_reasons = set()

            for reasons in (
                exclusion_reasons.values()
            ):
                all_reasons.update(
                    reasons
                )

            provisional_reason = (
                "NO_DISCOUNT_OPTIMAL"
            )

            for candidate_reason in (
                "DISCOUNT_FREQUENCY_LIMIT",
                "PROFIT_FLOOR_VIOLATION",
                "INSUFFICIENT_UPLIFT",
                "DISCOUNT_CAP_EXCEEDED",
            ):

                if candidate_reason in all_reasons:

                    provisional_reason = (
                        candidate_reason
                    )

                    break

    else:

        provisional_reason = "APPROVED"

    return (
        best,
        provisional_reason,
        checks,
        request.pred_optimal_discount,
        profit_map,
        uplift,
    )


# =============================================================================
# HUMAN APPROVAL GATE
# =============================================================================

def apply_approval_gate(
    candidate_discount: float,
    provisional_reason_code: str,
    config: PolicyConfig,
):

    if (
        candidate_discount
        > config.human_approval_threshold_percent
    ):

        return (
            "HUMAN_APPROVAL_REQUIRED",
            "HIGH_DISCOUNT_REQUIRES_APPROVAL",
        )

    return (
        "APPROVED",
        provisional_reason_code,
    )


# =============================================================================
# EXPLANATION
# =============================================================================

def _generate_explanation(
    reason_code: str,
    discount: float,
) -> str:

    templates = {

        "MODEL_UNAVAILABLE":
            "Safe fallback applied because the pricing "
            "model was unavailable.",

        "NO_DISCOUNT_OPTIMAL":
            "No discount approved because the predicted "
            "economic benefit does not justify reducing "
            "the merchant's margin.",

        "INSUFFICIENT_UPLIFT":
            "Discount rejected because the predicted "
            "incremental conversion lift does not meet "
            "the minimum required threshold.",

        "PROFIT_FLOOR_VIOLATION":
            "Discount rejected because it would push the "
            "merchant's margin below the configured "
            "profit floor.",

        "DISCOUNT_FREQUENCY_LIMIT":
            "Discount rejected because this customer has "
            "reached the configured discount frequency limit.",

        "DISCOUNT_CAP_EXCEEDED":
            f"Discount capped because {discount:g}% "
            "exceeds the maximum allowed by policy.",

        "HIGH_DISCOUNT_REQUIRES_APPROVAL":
            f"{discount:g}% discount requires human approval "
            "because it exceeds the automatic approval threshold.",

        "APPROVED":
            f"{discount:g}% discount approved because it "
            "provides sufficient incremental conversion lift "
            "while remaining above the merchant's profit floor.",
    }

    return templates.get(
        reason_code,
        f"Decision reason: {reason_code}; "
        f"discount: {discount:g}%.",
    )


# =============================================================================
# AUDIT RECORD
# =============================================================================

def build_audit_record(
    request: CheckoutDecisionRequest,
    config: PolicyConfig,
    *,
    decision: str,
    reason_code: str,
    policy_selected_discount: float,
    checks: list,
    model_recommended_discount,
    model_available: bool,
    profit_map: Optional[dict] = None,
    uplift: Optional[dict] = None,
) -> dict:

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # -------------------------------------------------------------------------
    # ALWAYS preserve the legacy audit fields.
    #
    # This is important because the existing API/tests expect them.
    # -------------------------------------------------------------------------

    if not model_available:

        pred_p0 = None
        pred_p5 = None
        pred_p10 = None
        pred_p15 = None

        pred_profit_0 = None
        pred_profit_5 = None
        pred_profit_10 = None
        pred_profit_15 = None

        expected_profit_selected = None
        predicted_uplift_selected = None

    else:

        predictions = get_prediction_map(
            request
        )

        profits = get_profit_map(
            request
        )

        pred_p0 = predictions.get(
            0.0
        )

        pred_p5 = predictions.get(
            5.0
        )

        pred_p10 = predictions.get(
            10.0
        )

        pred_p15 = predictions.get(
            15.0
        )

        pred_profit_0 = profits.get(
            0.0
        )

        pred_profit_5 = profits.get(
            5.0
        )

        pred_profit_10 = profits.get(
            10.0
        )

        pred_profit_15 = profits.get(
            15.0
        )

        if profit_map is not None:

            expected_profit_selected = (
                profit_map.get(
                    float(
                        policy_selected_discount
                    )
                )
            )

        else:

            expected_profit_selected = None

        if (
            policy_selected_discount == 0
        ):

            predicted_uplift_selected = 0.0

        elif uplift is not None:

            predicted_uplift_selected = (
                uplift.get(
                    float(
                        policy_selected_discount
                    )
                )
            )

        else:

            predicted_uplift_selected = None

    # -------------------------------------------------------------------------
    # Build audit record
    # -------------------------------------------------------------------------

    record = {

        "timestamp":
            timestamp,

        "customer_id":
            request.customer_id,

        "cart_value":
            request.cart_value,

        # Legacy audit fields
        "pred_p0":
            pred_p0,

        "pred_p5":
            pred_p5,

        "pred_p10":
            pred_p10,

        "pred_p15":
            pred_p15,

        "pred_expected_profit_0":
            pred_profit_0,

        "pred_expected_profit_5":
            pred_profit_5,

        "pred_expected_profit_10":
            pred_profit_10,

        "pred_expected_profit_15":
            pred_profit_15,

        # Continuous predictions
        "discount_predictions":
            (
                request.discount_predictions
                if model_available
                else None
            ),

        "expected_profit_predictions":
            (
                request.expected_profit_predictions
                if model_available
                else None
            ),

        "model_recommended_discount":
            model_recommended_discount,

        "policy_selected_discount":
            policy_selected_discount,

        "decision":
            decision,

        "reason_code":
            reason_code,

        "policy_checks":
            checks,

        "expected_profit_selected":
            expected_profit_selected,

        "predicted_uplift_selected":
            predicted_uplift_selected,

        "model_version":
            MODEL_VERSION,

        "policy_version":
            config.version,

        "model_available":
            model_available,

        "explanation":
            _generate_explanation(
                reason_code,
                policy_selected_discount,
            ),
    }

    return record


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def evaluate_policy(
    request: Union[
        CheckoutDecisionRequest,
        dict,
    ],
    config: Union[
        PolicyConfig,
        dict,
        None,
    ] = None,
) -> dict:

    request = _coerce_request(
        request
    )

    config = _coerce_config(
        config
    )

    # -------------------------------------------------------------------------
    # RULE 1 — MODEL AVAILABILITY
    # -------------------------------------------------------------------------

    valid_model, model_detail = (
        validate_model_output(
            request
        )
    )

    if not valid_model:

        checks = [
            {
                "check":
                    "model_availability",

                "discount":
                    None,

                "passed":
                    False,

                "detail":
                    model_detail,
            }
        ]

        return build_audit_record(
            request,
            config,
            decision="FALLBACK",
            reason_code="MODEL_UNAVAILABLE",
            policy_selected_discount=(
                config.fallback_discount_percent
            ),
            checks=checks,
            model_recommended_discount=None,
            model_available=False,
        )

    # -------------------------------------------------------------------------
    # Candidate selection
    # -------------------------------------------------------------------------

    (
        best,
        provisional_reason,
        checks,
        model_recommended,
        profit_map,
        uplift,
    ) = select_best_valid_discount(
        request,
        config,
    )

    checks = [
        {
            "check":
                "model_availability",

            "discount":
                None,

            "passed":
                True,

            "detail":
                model_detail,
        }
    ] + checks

    # -------------------------------------------------------------------------
    # Human approval gate
    # -------------------------------------------------------------------------

    decision, reason_code = (
        apply_approval_gate(
            best,
            provisional_reason,
            config,
        )
    )

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------

    return build_audit_record(
        request,
        config,

        decision=decision,

        reason_code=reason_code,

        policy_selected_discount=best,

        checks=checks,

        model_recommended_discount=
            model_recommended,

        model_available=True,

        profit_map=profit_map,

        uplift=uplift,
    )