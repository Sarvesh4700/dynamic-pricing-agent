"""
src/agent_tools.py

Deterministic tools exposed to the pricing agent, plus the session-scoped
executor that enforces the mandatory call order.

    score_checkout  ->  evaluate_policy  ->  record_decision

DESIGN NOTE - why the tools take (almost) no arguments
------------------------------------------------------
score_checkout and evaluate_policy accept an EMPTY argument object. The checkout
features are bound to the ToolSession by the caller, not passed by the model, so
the LLM has no channel through which to alter cart value, margin, probabilities
or profits. record_decision accepts exactly one string: the merchant-facing
explanation. Every number, version and reason code in the audit row comes from
deterministic session state.

Nothing here trains anything. Scoring loads the already-persisted T-learner
pipelines from models/uplift/ and reuses src/uplift_model.py's own profit
functions, so the economics stay defined in one place.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime, timezone
from typing import Optional, Union

from src import policy_engine
from src.policy_engine import (
    DISCOUNT_LEVELS,
    MODEL_VERSION,
    CheckoutDecisionRequest,
    PolicyConfig,
    validate_model_output,
)


DEFAULT_MODEL_DIR = "models/uplift"
DEFAULT_SCORING_TIMEOUT_SECONDS = 30.0


# Mirrors CATEGORICAL_FEATURES + NUMERICAL_FEATURES in src/uplift_model.py.
# ModelScorer asserts agreement against that module at scoring time, so a drift
# is a loud failure rather than a silently wrong prediction.
MODEL_FEATURE_FIELDS = (
    "customer_type",
    "category",
    "device_type",
    "previous_purchases",
    "previous_abandons",
    "days_since_last_purchase",
    "customer_lifetime_value",
    "historical_discount_rate",
    "cart_value",
    "items_count",
    "margin_percentage",
    "time_on_checkout_seconds",
    "pages_viewed",
    "payment_attempts",
    "hour",
    "day_of_week",
)


class ModelScoringError(Exception):
    """Scoring could not produce usable predictions."""


class ModelScoringTimeout(ModelScoringError):
    """Scoring exceeded its wall-clock budget (the '2 AM timeout' scenario)."""


class CheckoutValidationError(Exception):
    """The inbound checkout payload is unusable."""


# --------------------------------------------------------------------------------------
# CHECKOUT REQUEST
# --------------------------------------------------------------------------------------
@dataclass
class CheckoutRequest:
    """Everything observable at checkout time.

    Superset of policy_engine.CheckoutDecisionRequest: the T-learner also needs
    customer_lifetime_value, items_count and days_since_last_purchase, which the
    policy engine has no use for.
    """

    # identity
    customer_id: str
    transaction_id: str

    # customer
    customer_type: str
    previous_purchases: int
    previous_abandons: int
    days_since_last_purchase: float
    customer_lifetime_value: float
    historical_discount_rate: float
    recent_discount_count: int
    days_since_last_discount: Optional[float]

    # cart / session
    cart_value: float
    items_count: int
    category: str
    margin_percentage: float
    device_type: str
    time_on_checkout_seconds: float
    pages_viewed: int
    payment_attempts: int
    hour: int
    day_of_week: int

    def to_model_features(self) -> dict:
        return {
            name: getattr(self, name)
            for name in MODEL_FEATURE_FIELDS
        }

    def to_policy_request(
        self,
        model_output: Optional[dict],
    ) -> CheckoutDecisionRequest:
        """Project onto the policy engine's dataclass, attaching ML fields when
        they exist. Missing/None predictions deliberately fall through to the
        policy engine's own RULE 1 fallback.
        """
        allowed = {
            f.name
            for f in dataclass_fields(CheckoutDecisionRequest)
        }

        payload = {
            k: v
            for k, v in self.to_dict().items()
            if k in allowed
        }

        for key in (
            "pred_p0",
            "pred_p5",
            "pred_p10",
            "pred_p15",
            "pred_expected_profit_0",
            "pred_expected_profit_5",
            "pred_expected_profit_10",
            "pred_expected_profit_15",
            "pred_optimal_discount",
            "pred_max_expected_profit",
        ):
            payload[key] = (model_output or {}).get(key)

        return CheckoutDecisionRequest(**payload)

    def to_dict(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in dataclass_fields(self)
        }


_REQUIRED_NUMERIC = (
    "previous_purchases",
    "previous_abandons",
    "days_since_last_purchase",
    "customer_lifetime_value",
    "historical_discount_rate",
    "recent_discount_count",
    "cart_value",
    "items_count",
    "margin_percentage",
    "time_on_checkout_seconds",
    "pages_viewed",
    "payment_attempts",
    "hour",
    "day_of_week",
)

_REQUIRED_STRINGS = (
    "customer_id",
    "transaction_id",
    "customer_type",
    "category",
    "device_type",
)


def coerce_checkout(
    payload: Union[CheckoutRequest, dict],
) -> CheckoutRequest:
    """Validate + coerce an inbound checkout. Raises CheckoutValidationError
    with every problem listed at once, so a caller sees the whole picture.
    """
    if isinstance(payload, CheckoutRequest):
        checkout = payload

    elif isinstance(payload, dict):
        known = {
            f.name
            for f in dataclass_fields(CheckoutRequest)
        }

        missing = sorted(known - set(payload))
        if missing:
            raise CheckoutValidationError(
                f"missing checkout field(s): {', '.join(missing)}"
            )

        unknown = sorted(set(payload) - known)
        if unknown:
            raise CheckoutValidationError(
                f"unrecognized checkout field(s): {', '.join(unknown)}"
            )

        checkout = CheckoutRequest(**payload)

    else:
        raise CheckoutValidationError(
            "checkout must be a CheckoutRequest or a dict"
        )

    problems = []

    for name in _REQUIRED_STRINGS:
        value = getattr(checkout, name)

        if not isinstance(value, str) or not value.strip():
            problems.append(
                f"'{name}' must be a non-empty string"
            )

    for name in _REQUIRED_NUMERIC:
        value = getattr(checkout, name)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(
                f"'{name}' must be numeric (got {type(value).__name__})"
            )
            continue

        if value != value or value in (
            float("inf"),
            float("-inf"),
        ):
            problems.append(
                f"'{name}' must be finite"
            )

    if not problems:
        if checkout.cart_value <= 0:
            problems.append("'cart_value' must be positive")

        if not (0 <= checkout.margin_percentage <= 100):
            problems.append(
                "'margin_percentage' must be within [0, 100]"
            )

        if not (0 <= checkout.hour <= 23):
            problems.append(
                "'hour' must be within [0, 23]"
            )

        if not (0 <= checkout.day_of_week <= 6):
            problems.append(
                "'day_of_week' must be within [0, 6]"
            )

    if problems:
        raise CheckoutValidationError("; ".join(problems))

    return checkout


# --------------------------------------------------------------------------------------
# TOOL 1 - SCORING
# --------------------------------------------------------------------------------------
def _env_flag(name: str) -> bool:
    return str(
        os.getenv(name, "")
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _simulate_mode(explicit) -> Optional[str]:
    """None | 'error' | 'timeout'. Explicit constructor arg wins over the env."""
    if explicit is not None:
        if explicit is False:
            return None

        if explicit is True:
            return "error"

        return str(explicit).lower() or None

    raw = str(
        os.getenv("SIMULATE_MODEL_FAILURE", "")
    ).strip().lower()

    if raw in {
        "1",
        "true",
        "yes",
        "on",
        "error",
    }:
        return "error"

    if raw in {
        "timeout",
        "hang",
        "slow",
    }:
        return "timeout"

    return None


class ModelScorer:
    """Loads the persisted T-learner arms once and scores single checkouts.

    Failure is injectable two ways: SIMULATE_MODEL_FAILURE=true|timeout in the
    environment (for the demo), or simulate_failure=... in the constructor (for
    tests). Both routes lead to the same MODEL_UNAVAILABLE audit outcome.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        timeout_seconds: Optional[float] = None,
        simulate_failure=None,
    ):
        self.model_dir = model_dir

        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv(
                "MODEL_SCORING_TIMEOUT_SECONDS",
                DEFAULT_SCORING_TIMEOUT_SECONDS,
            )
        )

        self._simulate_failure = simulate_failure
        self._models = None
        self.model_version = MODEL_VERSION

    # -- loading ----------------------------------------------------------------
    def _load_models(self):
        if self._models is not None:
            return self._models

        try:
            import joblib
        except ImportError as exc:
            raise ModelScoringError(
                "joblib is unavailable, cannot load model artifacts"
            ) from exc

        models = {}

        for d in DISCOUNT_LEVELS:
            path = os.path.join(
                self.model_dir,
                f"model_discount_{d}.joblib",
            )

            if not os.path.exists(path):
                raise ModelScoringError(
                    f"model artifact not found: {path} "
                    "(run 'python src/uplift_model.py' first)"
                )

            try:
                models[d] = joblib.load(path)
            except Exception as exc:
                raise ModelScoringError(
                    f"failed to load {path}: {type(exc).__name__}"
                ) from exc

        self._models = models
        return models

    # -- scoring ----------------------------------------------------------------
    def score(self, checkout: CheckoutRequest) -> dict:
        mode = _simulate_mode(self._simulate_failure)

        if mode == "error":
            raise ModelScoringError(
                "injected model-service failure "
                "(SIMULATE_MODEL_FAILURE)"
            )

        if mode == "timeout":
            raise ModelScoringTimeout(
                f"injected model-service timeout after "
                f"{self.timeout_seconds:.1f}s "
                f"(SIMULATE_MODEL_FAILURE=timeout)"
            )

        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            future = executor.submit(
                self._score_now,
                checkout,
            )

            result = future.result(
                timeout=self.timeout_seconds
            )

        except FuturesTimeout as exc:
            raise ModelScoringTimeout(
                f"model scoring exceeded "
                f"{self.timeout_seconds:.1f}s"
            ) from exc

        except ModelScoringError:
            raise

        except Exception as exc:
            raise ModelScoringError(
                f"{type(exc).__name__} during model scoring"
            ) from exc

        finally:
            executor.shutdown(wait=False)

        result["scoring_latency_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return result

    def _score_now(self, checkout: CheckoutRequest) -> dict:
        try:
            import pandas as pd

            from src.uplift_model import (
                FEATURE_COLS,
                calculate_expected_profit,
                calculate_uplift,
                predict_potential_outcomes,
            )

        except ImportError as exc:
            raise ModelScoringError(
                f"scoring dependencies unavailable "
                f"({type(exc).__name__})"
            ) from exc

        features = checkout.to_model_features()

        drift = set(FEATURE_COLS) - set(features)

        if drift:
            raise ModelScoringError(
                f"checkout schema is missing model feature(s): "
                f"{sorted(drift)}"
            )

        models = self._load_models()

        df = pd.DataFrame([features])

        df = predict_potential_outcomes(
            models,
            df,
        )

        df = calculate_uplift(df)

        # Reused verbatim from src/uplift_model.py - the profit definition is
        # never re-implemented in the agent layer.
        df = calculate_expected_profit(
            df,
            prob_prefix="pred_p",
            profit_prefix="pred_expected_profit",
        )

        row = df.iloc[0]

        out = {
            f"pred_p{d}": float(row[f"pred_p{d}"])
            for d in DISCOUNT_LEVELS
        }

        out.update({
            f"pred_expected_profit_{d}": float(
                row[f"pred_expected_profit_{d}"]
            )
            for d in DISCOUNT_LEVELS
        })

        out.update({
            "pred_uplift_5": float(row["pred_uplift_5"]),
            "pred_uplift_10": float(row["pred_uplift_10"]),
            "pred_uplift_15": float(row["pred_uplift_15"]),
            "pred_optimal_discount": int(
                row["pred_optimal_discount"]
            ),
            "pred_max_expected_profit": float(
                row["pred_max_expected_profit"]
            ),
            "model_version": self.model_version,
            "scored_at": datetime.now(
                timezone.utc
            ).isoformat(),
        })

        return out


# --------------------------------------------------------------------------------------
# TOOL SCHEMAS (Anthropic tool-use format)
# --------------------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "score_checkout",
        "description": (
            "Run the trained uplift (T-learner) models on the checkout currently "
            "under review and return predicted conversion probabilities and "
            "expected profit for each candidate discount (0/5/10/15). Takes no "
            "arguments: the checkout is already bound to this session, so you "
            "must not and cannot supply feature values. Call this FIRST. If it "
            "reports an error, continue to evaluate_policy anyway - the policy "
            "engine owns the safe fallback."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "evaluate_policy",
        "description": (
            "Apply the merchant's deterministic policy engine to the model output "
            "from score_checkout and return the authoritative decision, selected "
            "discount, reason code and per-rule checks. Takes no arguments. Must "
            "be called after score_checkout and before you state any decision. "
            "Its verdict is final."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "record_decision",
        "description": (
            "Persist the completed decision to the audit trail. Supply only your "
            "concise merchant-facing explanation; all numbers, versions and "
            "reason codes are taken from the deterministic tool outputs, never "
            "from you. Call this last."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_explanation": {
                    "type": "string",
                    "description": (
                        "One to three sentences a merchant can read, grounded "
                        "strictly in the policy and model tool outputs. No hidden "
                        "reasoning, no invented figures."
                    ),
                }
            },
            "required": ["agent_explanation"],
            "additionalProperties": False,
        },
    },
]


TOOL_NAMES = tuple(
    t["name"]
    for t in TOOL_SCHEMAS
)


# --------------------------------------------------------------------------------------
# SESSION-SCOPED TOOL EXECUTOR
# --------------------------------------------------------------------------------------
class ToolSession:
    """Holds all mutable state for one checkout decision and executes the tools.

    Enforces ordering (RULE: policy cannot run before scoring was attempted;
    audit cannot run before policy produced a verdict) and is idempotent for
    record_decision so a chatty model cannot create duplicate audit rows.
    """

    def __init__(
        self,
        checkout: CheckoutRequest,
        *,
        scorer,
        policy_config: PolicyConfig,
        audit_store,
        agent_model: str,
    ):
        self.checkout = checkout
        self.scorer = scorer
        self.policy_config = policy_config
        self.audit_store = audit_store
        self.agent_model = agent_model

        self.score_attempted = False
        self.model_output: Optional[dict] = None
        self.policy_output: Optional[dict] = None
        self.audit_id: Optional[int] = None
        self.error_type: Optional[str] = None
        self.tool_trace: list = []

    # -- dispatch ---------------------------------------------------------------
    def execute(
        self,
        name: str,
        arguments: Optional[dict] = None,
    ) -> dict:
        arguments = arguments or {}

        handler = {
            "score_checkout": self._tool_score_checkout,
            "evaluate_policy": self._tool_evaluate_policy,
            "record_decision": self._tool_record_decision,
        }.get(name)

        if handler is None:
            result = {
                "ok": False,
                "error": "UNKNOWN_TOOL",
                "detail": (
                    f"no such tool '{name}'; "
                    f"available: {list(TOOL_NAMES)}"
                ),
            }

        else:
            try:
                result = handler(arguments)

            except Exception as exc:
                # a tool must never break the orchestrator
                result = {
                    "ok": False,
                    "error": "TOOL_EXECUTION_ERROR",
                    "detail": f"{type(exc).__name__}: {exc}",
                }

        self.tool_trace.append({
            "tool": name,
            "ok": bool(result.get("ok")),
            "error": result.get("error"),
        })

        return result

    # -- tool 1 -----------------------------------------------------------------
    def _tool_score_checkout(
        self,
        _arguments: dict,
    ) -> dict:
        if self.model_output is not None:
            return {
                "ok": True,
                "cached": True,
                "model_output": self.model_output,
            }

        self.score_attempted = True

        try:
            self.model_output = self.scorer.score(
                self.checkout
            )

        except ModelScoringTimeout as exc:
            self.error_type = "MODEL_TIMEOUT"

            return {
                "ok": False,
                "error": "MODEL_TIMEOUT",
                "detail": str(exc),
                "next_step": (
                    "call evaluate_policy; it will apply "
                    "the safe fallback"
                ),
            }

        except ModelScoringError as exc:
            self.error_type = "MODEL_SCORING_ERROR"

            return {
                "ok": False,
                "error": "MODEL_UNAVAILABLE",
                "detail": str(exc),
                "next_step": (
                    "call evaluate_policy; it will apply "
                    "the safe fallback"
                ),
            }

        return {
            "ok": True,
            "model_output": self.model_output,
        }

    # -- tool 2 -----------------------------------------------------------------
    def _tool_evaluate_policy(
        self,
        _arguments: dict,
    ) -> dict:
        if self.policy_output is not None:
            return {
                "ok": True,
                "cached": True,
                "policy_output": self.policy_output,
            }

        if not self.score_attempted:
            return {
                "ok": False,
                "error": "OUT_OF_ORDER",
                "detail": (
                    "score_checkout must be attempted "
                    "before evaluate_policy"
                ),
            }

        if _env_flag("SIMULATE_POLICY_FAILURE"):
            self.policy_output = self._policy_engine_unavailable(
                "injected policy-engine failure "
                "(SIMULATE_POLICY_FAILURE)"
            )

            self.error_type = "POLICY_ENGINE_ERROR"

            return {
                "ok": False,
                "error": "POLICY_ENGINE_UNAVAILABLE",
                "policy_output": self.policy_output,
            }

        try:
            request = self.checkout.to_policy_request(
                self.model_output
            )

            self.policy_output = policy_engine.evaluate_policy(
                request,
                self.policy_config,
            )

        except Exception as exc:
            self.policy_output = self._policy_engine_unavailable(
                f"{type(exc).__name__}: {exc}"
            )

            self.error_type = "POLICY_ENGINE_ERROR"

            return {
                "ok": False,
                "error": "POLICY_ENGINE_UNAVAILABLE",
                "policy_output": self.policy_output,
            }

        # Distinguish "no predictions at all" from "predictions arrived malformed";
        # both end in MODEL_UNAVAILABLE, but the audit trail should say which.
        if (
            self.model_output is not None
            and self.policy_output.get("reason_code")
            == "MODEL_UNAVAILABLE"
        ):
            valid, detail = validate_model_output(
                self.checkout.to_policy_request(
                    self.model_output
                )
            )

            if not valid:
                self.error_type = "MALFORMED_MODEL_OUTPUT"

                self.policy_output[
                    "malformed_model_output_detail"
                ] = detail

        return {
            "ok": True,
            "policy_output": self.policy_output,
        }

    def _policy_engine_unavailable(
        self,
        detail: str,
    ) -> dict:
        """Last-resort record when the policy engine itself cannot answer.
        The discount is the configured fallback (0%), never a guess.
        """
        fallback = (
            getattr(
                self.policy_config,
                "fallback_discount_percent",
                0,
            )
            or 0
        )

        return {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "customer_id": self.checkout.customer_id,
            "cart_value": self.checkout.cart_value,
            "model_recommended_discount": (
                self.model_output or {}
            ).get("pred_optimal_discount"),
            "policy_selected_discount": int(fallback),
            "decision": "FALLBACK",
            "reason_code": "POLICY_ENGINE_UNAVAILABLE",
            "policy_checks": [
                {
                    "check": "policy_engine_availability",
                    "discount": None,
                    "passed": False,
                    "detail": detail,
                }
            ],
            "expected_profit_selected": None,
            "predicted_uplift_selected": None,
            "model_version": MODEL_VERSION,
            "policy_version": getattr(
                self.policy_config,
                "version",
                "unknown",
            ),
            "model_available": self.model_output is not None,
            "explanation": (
                "Safe fallback applied because the policy engine "
                "could not be evaluated for this checkout."
            ),
        }

    # -- tool 3 -----------------------------------------------------------------
    def _tool_record_decision(
        self,
        arguments: dict,
    ) -> dict:
        if self.policy_output is None:
            return {
                "ok": False,
                "error": "OUT_OF_ORDER",
                "detail": (
                    "evaluate_policy must produce a verdict before "
                    "record_decision"
                ),
            }

        if self.audit_id is not None:
            return {
                "ok": True,
                "cached": True,
                "audit_id": self.audit_id,
            }

        explanation = arguments.get(
            "agent_explanation"
        )

        policy_explanation = (
            self.policy_output.get("explanation")
            or "Checkout continued using the configured safe fallback discount."
        )

        # The policy engine is authoritative. If the LLM explanation
        # conflicts with the policy verdict, never write the conflicting
        # explanation to the audit trail.
        from src.agent import (
            explanation_conflicts_with_policy,
            sanitize_explanation,
        )

        candidate = sanitize_explanation(
            explanation
        )

        if (
            not candidate
            or explanation_conflicts_with_policy(
                candidate,
                decision=self.policy_output.get(
                    "decision",
                    "FALLBACK",
                ),
                discount=self.policy_output.get(
                    "policy_selected_discount",
                    0,
                ),
            )
        ):
            explanation = policy_explanation
            explanation_source = (
                "policy_engine_llm_rejected"
            )

        else:
            explanation = candidate
            explanation_source = "llm"

        audit_id = self.write_audit(
            explanation,
            explanation_source=explanation_source,
        )

        return {
            "ok": True,
            "audit_id": audit_id,
        }

    # -- called by the orchestrator too ----------------------------------------
    def write_audit(
        self,
        explanation: Optional[str],
        *,
        explanation_source: str,
        latency_ms: Optional[float] = None,
    ) -> Optional[int]:
        """Deterministic audit write. Idempotent per session."""

        if self.audit_id is not None:
            return self.audit_id

        policy_output = self.policy_output or {}

        text = (
            explanation
            or policy_output.get("explanation")
            or ""
        )

        success = policy_output.get(
            "decision"
        ) in {
            "APPROVED",
            "HUMAN_APPROVAL_REQUIRED",
        }

        if self.audit_store is None:
            return None

        self.audit_id = self.audit_store.record_decision(
            transaction_id=self.checkout.transaction_id,
            customer_id=self.checkout.customer_id,
            input_features=self.checkout.to_dict(),
            model_output=self.model_output,
            policy_output=policy_output,
            final_decision=policy_output.get(
                "decision",
                "FALLBACK",
            ),
            final_discount=policy_output.get(
                "policy_selected_discount",
                0,
            ),
            reason_code=policy_output.get(
                "reason_code",
                "MODEL_UNAVAILABLE",
            ),
            agent_explanation=text,
            model_version=policy_output.get(
                "model_version",
                MODEL_VERSION,
            ),
            policy_version=policy_output.get(
                "policy_version",
                getattr(
                    self.policy_config,
                    "version",
                    "unknown",
                ),
            ),
            agent_model=self.agent_model,
            success=success,
            error_type=self.error_type,
            explanation_source=explanation_source,
            latency_ms=latency_ms,
        )

        return self.audit_id


def tool_result_to_json(result: dict) -> str:
    """Compact, safe serialization of a tool result for the LLM transcript."""
    try:
        return json.dumps(
            result,
            default=str,
        )
    except Exception:
        return json.dumps({
            "ok": False,
            "error": "UNSERIALIZABLE_TOOL_RESULT",
        })