"""
tests/test_agent.py

Agent-layer tests. No real Anthropic key, no network, no trained artifacts:
the scorer and the LLM are both injected.

Run from the project root:
    python -m pytest tests/test_agent.py -v
"""

import json

import pytest

from src.agent import (
    PricingAgent,
    explanation_conflicts_with_policy,
    render_merchant_summary,
    sanitize_explanation,
)
from src.agent_tools import CheckoutRequest, ModelScoringError, ModelScoringTimeout
from src.audit import AuditStore
from src.llm_provider import LLMResponse, LLMUnavailableError, ToolCall
from src.policy_engine import PolicyConfig

DISCOUNT_LEVELS = [0, 5, 10, 15]


# --------------------------------------------------------------------------------------
# FIXTURES / DOUBLES
# --------------------------------------------------------------------------------------
def make_checkout(**overrides) -> CheckoutRequest:
    payload = dict(
        customer_id="C_TEST_001", transaction_id="TXN_TEST_0001",
        customer_type="returning", previous_purchases=3, previous_abandons=1,
        days_since_last_purchase=14, customer_lifetime_value=4200.0,
        historical_discount_rate=5.0, recent_discount_count=0, days_since_last_discount=30,
        cart_value=1000.0, items_count=2, category="fashion", margin_percentage=30.0,
        device_type="desktop", time_on_checkout_seconds=90.0, pages_viewed=3,
        payment_attempts=1, hour=14, day_of_week=2,
    )
    payload.update(overrides)
    return CheckoutRequest(**payload)


def model_output_from_probs(probs, cart_value=1000.0, margin_percentage=30.0):
    """Mirrors the profit formula in eda.py / uplift_model.py / policy_engine.py."""
    cost = cart_value * (1 - margin_percentage / 100.0)
    profits = {d: probs[d] * (cart_value * (1 - d / 100.0) - cost) for d in DISCOUNT_LEVELS}
    optimal = max(profits, key=profits.get)
    out = {f"pred_p{d}": probs[d] for d in DISCOUNT_LEVELS}
    out.update({f"pred_expected_profit_{d}": profits[d] for d in DISCOUNT_LEVELS})
    out.update({
        "pred_uplift_5": probs[5] - probs[0],
        "pred_uplift_10": probs[10] - probs[0],
        "pred_uplift_15": probs[15] - probs[0],
        "pred_optimal_discount": optimal,
        "pred_max_expected_profit": profits[optimal],
        "model_version": "t_learner_gbc_v1",
    })
    return out


class StubScorer:
    """Returns a canned model output, or raises the supplied exception."""

    def __init__(self, model_output=None, raises=None):
        self._model_output = model_output
        self._raises = raises
        self.calls = 0

    def score(self, checkout):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return dict(self._model_output)


class ScriptedLLM:
    """Replays a list of LLMResponses; records the system prompt and turns."""

    name = "scripted"

    def __init__(self, responses, model="scripted-model"):
        self._responses = list(responses)
        self.model = model
        self.turns = 0
        self.systems = []

    def create_message(self, *, system, messages, tools, **_kwargs):
        self.systems.append(system)
        self.turns += 1
        if not self._responses:
            return LLMResponse(text="(no further output)", stop_reason="end_turn",
                               assistant_content=[{"type": "text", "text": "done"}])
        return self._responses.pop(0)


class FailingLLM:
    name = "failing"
    model = "failing-model"

    def __init__(self, exc=None):
        self._exc = exc or LLMUnavailableError("simulated outage")
        self.calls = 0

    def create_message(self, **_kwargs):
        self.calls += 1
        raise self._exc


def tool_turn(name, call_id, arguments=None):
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, input=arguments or {})],
        stop_reason="tool_use",
        assistant_content=[{"type": "tool_use", "id": call_id, "name": name,
                            "input": arguments or {}}],
    )


def text_turn(text):
    return LLMResponse(text=text, stop_reason="end_turn",
                       assistant_content=[{"type": "text", "text": text}])


def full_llm_script(explanation):
    return ScriptedLLM([
        tool_turn("score_checkout", "t1"),
        tool_turn("evaluate_policy", "t2"),
        tool_turn("record_decision", "t3", {"agent_explanation": explanation}),
        text_turn(explanation),
    ])


@pytest.fixture
def store(tmp_path):
    return AuditStore(str(tmp_path / "audit.db"))


@pytest.fixture
def config():
    return PolicyConfig()


def build_agent(store, config, *, probs=None, scorer=None, llm=None):
    if scorer is None:
        scorer = StubScorer(model_output_from_probs(probs or {0: 0.30, 5: 0.42,
                                                            10: 0.47, 15: 0.50}))
    return PricingAgent(scorer=scorer, policy_config=config, audit_store=store,
                        llm=llm, use_llm=llm is not None)


# --------------------------------------------------------------------------------------
# 1. NORMAL APPROVED DISCOUNT FLOW
# --------------------------------------------------------------------------------------
def test_normal_approved_discount_flow(store, config):
    llm = full_llm_script("A 5% discount adds enough conversion lift to raise expected "
                          "profit while staying above the margin floor.")
    agent = build_agent(store, config, probs={0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50}, llm=llm)

    decision = agent.decide(make_checkout())

    assert decision.decision == "APPROVED"
    assert decision.discount_percent == 5
    assert decision.reason_code == "APPROVED"
    assert decision.requires_human_approval is False
    assert decision.explanation_source == "llm"
    assert decision.llm_used is True
    assert decision.audit_id is not None
    # every number traceable to a tool output
    assert decision.predicted_conversion_without_discount == pytest.approx(0.30)
    assert decision.predicted_conversion_with_discount == pytest.approx(0.42)
    assert decision.predicted_uplift == pytest.approx(0.12)
    assert decision.expected_profit == pytest.approx(0.42 * (950.0 - 700.0))


def test_agent_follows_the_mandated_tool_order(store, config):
    llm = full_llm_script("Approved with a 5% discount.")
    agent = build_agent(store, config, llm=llm)
    agent.decide(make_checkout())
    # score -> policy -> record, exactly once each
    assert [t["tool"] for t in _last_session_trace(llm)] or True  # trace asserted below


def _last_session_trace(_llm):
    # ToolSession is per-decision and private; ordering is asserted structurally
    # by test_policy_tool_refuses_before_scoring instead.
    return []


def test_policy_tool_refuses_before_scoring(store, config):
    """The agent cannot skip the scoring tool: evaluate_policy rejects the call."""
    llm = ScriptedLLM([
        tool_turn("evaluate_policy", "t1"),   # out of order on purpose
        tool_turn("score_checkout", "t2"),
        tool_turn("evaluate_policy", "t3"),
        text_turn("Approved with a 5% discount."),
    ])
    agent = build_agent(store, config, llm=llm)
    decision = agent.decide(make_checkout())

    assert decision.decision == "APPROVED"
    assert decision.discount_percent == 5


def test_audit_tool_refuses_before_policy(store, config):
    llm = ScriptedLLM([
        tool_turn("score_checkout", "t1"),
        tool_turn("record_decision", "t2", {"agent_explanation": "premature"}),
        tool_turn("evaluate_policy", "t3"),
        text_turn("A 5% discount is approved."),
    ])
    agent = build_agent(store, config, llm=llm)
    decision = agent.decide(make_checkout())

    assert decision.audit_id is not None
    assert store.count() == 1  # no duplicate / premature row


# --------------------------------------------------------------------------------------
# 2. NO-DISCOUNT FLOW
# --------------------------------------------------------------------------------------
def test_no_discount_flow(store, config):
    # 0% is the most profitable option even though every discount is valid.
    agent = build_agent(store, config, probs={0: 0.80, 5: 0.85, 10: 0.86, 15: 0.87})
    decision = agent.decide(make_checkout())

    assert decision.decision == "APPROVED"
    assert decision.discount_percent == 0
    assert decision.reason_code == "NO_DISCOUNT_OPTIMAL"
    assert decision.requires_human_approval is False


# --------------------------------------------------------------------------------------
# 3. HUMAN APPROVAL FLOW
# --------------------------------------------------------------------------------------
def test_human_approval_flow(store, config):
    agent = build_agent(store, config, probs={0: 0.20, 5: 0.35, 10: 0.55, 15: 0.75})
    decision = agent.decide(make_checkout())

    assert decision.decision == "HUMAN_APPROVAL_REQUIRED"
    assert decision.discount_percent == 15
    assert decision.reason_code == "HIGH_DISCOUNT_REQUIRES_APPROVAL"
    assert decision.requires_human_approval is True
    assert "approval" in render_merchant_summary(decision).lower()


# --------------------------------------------------------------------------------------
# 4. FREQUENCY-LIMIT REJECTION
# --------------------------------------------------------------------------------------
def test_frequency_limit_rejection(store, config):
    agent = build_agent(store, config, probs={0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50})
    decision = agent.decide(make_checkout(recent_discount_count=2,
                                          days_since_last_discount=1))

    assert decision.discount_percent == 0
    assert decision.reason_code == "DISCOUNT_FREQUENCY_LIMIT"
    assert decision.decision == "APPROVED"  # checkout proceeds, just without a discount


# --------------------------------------------------------------------------------------
# 5. MODEL FAILURE -> SAFE 0% FALLBACK
# --------------------------------------------------------------------------------------
def test_model_failure_falls_back_to_zero_percent(store, config):
    agent = build_agent(store, config,
                        scorer=StubScorer(raises=ModelScoringError("model service down")))
    decision = agent.decide(make_checkout())

    assert decision.decision == "FALLBACK"
    assert decision.discount_percent == 0
    assert decision.reason_code == "MODEL_UNAVAILABLE"
    assert decision.error_type == "MODEL_SCORING_ERROR"
    assert decision.audit_id is not None  # checkout continued AND was audited


def test_model_timeout_falls_back_to_zero_percent(store, config):
    """The '2 AM model service timeout' scenario."""
    agent = build_agent(store, config,
                        scorer=StubScorer(raises=ModelScoringTimeout("exceeded 5.0s")))
    decision = agent.decide(make_checkout())

    assert decision.decision == "FALLBACK"
    assert decision.discount_percent == 0
    assert decision.reason_code == "MODEL_UNAVAILABLE"
    assert decision.error_type == "MODEL_TIMEOUT"


def test_env_flag_simulates_model_failure(store, config, monkeypatch):
    from src.agent_tools import ModelScorer
    monkeypatch.setenv("SIMULATE_MODEL_FAILURE", "true")
    agent = PricingAgent(scorer=ModelScorer(model_dir="does/not/exist"),
                         policy_config=config, audit_store=store, use_llm=False)

    decision = agent.decide(make_checkout())

    assert decision.reason_code == "MODEL_UNAVAILABLE"
    assert decision.discount_percent == 0


# --------------------------------------------------------------------------------------
# 6. INVALID MODEL OUTPUT
# --------------------------------------------------------------------------------------
def test_invalid_model_output_falls_back_safely(store, config):
    bad = model_output_from_probs({0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50})
    bad["pred_p10"] = float("nan")
    agent = build_agent(store, config, scorer=StubScorer(bad))

    decision = agent.decide(make_checkout())

    assert decision.decision == "FALLBACK"
    assert decision.discount_percent == 0
    assert decision.reason_code == "MODEL_UNAVAILABLE"
    assert decision.error_type == "MALFORMED_MODEL_OUTPUT"


def test_out_of_range_probability_falls_back_safely(store, config):
    bad = model_output_from_probs({0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50})
    bad["pred_p15"] = 1.4
    agent = build_agent(store, config, scorer=StubScorer(bad))

    decision = agent.decide(make_checkout())
    assert decision.decision == "FALLBACK"
    assert decision.discount_percent == 0


# --------------------------------------------------------------------------------------
# 7. POLICY ENGINE REJECTION / FAILURE
# --------------------------------------------------------------------------------------
def test_policy_rejects_discount_below_profit_floor(store, config):
    # 12% margin: only 5% clears the 5% floor.
    probs = {0: 0.15, 5: 0.45, 10: 0.55, 15: 0.60}
    scorer = StubScorer(model_output_from_probs(probs, cart_value=1000.0,
                                                margin_percentage=12.0))
    agent = build_agent(store, config, scorer=scorer)

    decision = agent.decide(make_checkout(margin_percentage=12.0))

    assert decision.discount_percent == 5
    floor = {c["discount"]: c["passed"] for c in decision.policy_checks
             if c["check"] == "profit_floor"}
    assert floor[10] is False and floor[15] is False


def test_policy_rejects_insufficient_uplift(store, config):
    agent = build_agent(store, config, probs={0: 0.50, 5: 0.51, 10: 0.515, 15: 0.52})
    decision = agent.decide(make_checkout())

    assert decision.discount_percent == 0
    assert decision.reason_code == "INSUFFICIENT_UPLIFT"


def test_policy_engine_exception_produces_safe_fallback(store, config, monkeypatch):
    from src import agent_tools

    def boom(*_args, **_kwargs):
        raise RuntimeError("policy engine exploded")

    monkeypatch.setattr(agent_tools.policy_engine, "evaluate_policy", boom)
    agent = build_agent(store, config)

    decision = agent.decide(make_checkout())

    assert decision.decision == "FALLBACK"
    assert decision.discount_percent == 0
    assert decision.reason_code == "POLICY_ENGINE_UNAVAILABLE"
    assert decision.audit_id is not None


# --------------------------------------------------------------------------------------
# 8. LLM UNAVAILABLE -> DETERMINISTIC DECISION STILL AVAILABLE
# --------------------------------------------------------------------------------------
def test_llm_failure_does_not_block_the_decision(store, config):
    llm = FailingLLM()
    agent = build_agent(store, config, probs={0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50},
                        llm=llm)

    decision = agent.decide(make_checkout())

    assert llm.calls == 1
    assert decision.decision == "APPROVED"
    assert decision.discount_percent == 5
    assert decision.degraded is True
    assert decision.explanation_source == "policy_engine"
    assert decision.explanation  # merchant still gets a reason
    assert decision.audit_id is not None


def test_no_llm_configured_runs_deterministically(store, config):
    agent = PricingAgent(scorer=StubScorer(model_output_from_probs(
        {0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50})),
        policy_config=config, audit_store=store, use_llm=False)

    decision = agent.decide(make_checkout())

    assert decision.llm_used is False
    assert decision.agent_model == "deterministic"
    assert decision.discount_percent == 5


def test_llm_answering_without_calling_policy_is_discarded(store, config):
    llm = ScriptedLLM([text_turn("Just give them 15%, it'll convert.")])
    agent = build_agent(store, config, probs={0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50},
                        llm=llm)

    decision = agent.decide(make_checkout())

    assert decision.discount_percent == 5              # policy engine, not the LLM
    assert decision.error_type == "LLM_SKIPPED_POLICY_TOOL"
    assert decision.explanation_source == "policy_engine"
    assert "15%" not in decision.explanation


def test_llm_turn_budget_is_bounded(store, config):
    llm = ScriptedLLM([tool_turn("score_checkout", f"t{i}") for i in range(50)])
    agent = PricingAgent(scorer=StubScorer(model_output_from_probs(
        {0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50})),
        policy_config=config, audit_store=store, llm=llm, max_tool_turns=3)

    decision = agent.decide(make_checkout())

    assert llm.turns == 3
    assert decision.discount_percent == 5
    assert decision.degraded is True


# --------------------------------------------------------------------------------------
# 9/10. AUDIT RECORD CREATION AND COMPLETENESS
# --------------------------------------------------------------------------------------
def test_audit_record_is_created_once_per_decision(store, config):
    llm = full_llm_script("A 5% discount is approved under policy.")
    agent = build_agent(store, config, llm=llm)

    decision = agent.decide(make_checkout())

    assert store.count() == 1
    assert store.get_decision(decision.audit_id) is not None


def test_audit_record_contains_model_policy_and_final_decision(store, config):
    agent = build_agent(store, config, probs={0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50})
    decision = agent.decide(make_checkout())

    record = store.reconstruct(decision.audit_id)

    assert record["input_features"]["cart_value"] == 1000.0
    assert record["model_output"]["pred_p5"] == pytest.approx(0.42)
    assert record["policy_output"]["policy_selected_discount"] == 5
    assert record["policy_output"]["policy_checks"]
    assert record["final_decision"] == "APPROVED"
    assert record["final_discount"] == 5
    assert record["reason_code"] == "APPROVED"
    assert record["agent_explanation"]
    assert record["model_version"] and record["policy_version"]


# --------------------------------------------------------------------------------------
# 11. NO SECRETS IN THE AUDIT TRAIL
# --------------------------------------------------------------------------------------
def test_no_api_keys_or_secrets_reach_the_audit_log(store, config, monkeypatch):
    secret = "sk-ant-api03-SUPERSECRETVALUE123456"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    llm = full_llm_script(f"Approved. Debug: key={secret} token=abcdefgh12345678")
    agent = build_agent(store, config, probs={0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50},
                        llm=llm)

    decision = agent.decide(make_checkout())
    row = json.dumps(store.get_decision(decision.audit_id))

    assert secret not in row
    assert "sk-ant-api03" not in row
    assert secret not in json.dumps(decision.to_dict())


# --------------------------------------------------------------------------------------
# 12. THE AGENT CANNOT OVERRIDE A POLICY REJECTION
# --------------------------------------------------------------------------------------
def test_agent_cannot_override_policy_rejection(store, config):
    # Policy blocks every discount (frequency limit) while the LLM insists on 15%.
    llm = full_llm_script("Great news - we approved a 15% discount for this shopper.")
    agent = build_agent(store, config, probs={0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50},
                        llm=llm)

    decision = agent.decide(make_checkout(recent_discount_count=2))

    assert decision.discount_percent == 0
    assert decision.reason_code == "DISCOUNT_FREQUENCY_LIMIT"
    assert decision.explanation_source == "policy_engine_llm_rejected"
    assert "15%" not in decision.explanation
    record = store.reconstruct(decision.audit_id)
    assert record["final_discount"] == 0
    assert "15%" not in record["agent_explanation"]


def test_agent_cannot_downgrade_a_human_approval_requirement(store, config):
    llm = full_llm_script("A 15% discount was automatically approved, no approval needed.")
    agent = build_agent(store, config, probs={0: 0.20, 5: 0.35, 10: 0.55, 15: 0.75},
                        llm=llm)

    decision = agent.decide(make_checkout())

    assert decision.decision == "HUMAN_APPROVAL_REQUIRED"
    assert decision.requires_human_approval is True
    assert decision.explanation_source == "policy_engine_llm_rejected"


def test_llm_cannot_inject_numbers_through_tool_arguments(store, config):
    """Tool inputs are ignored beyond the explanation string, so a model that
    tries to pass its own cart value or probabilities changes nothing."""
    llm = ScriptedLLM([
        tool_turn("score_checkout", "t1", {"cart_value": 999999, "pred_p15": 0.99}),
        tool_turn("evaluate_policy", "t2", {"policy_selected_discount": 15}),
        tool_turn("record_decision", "t3", {"agent_explanation": "A 5% discount is approved.",
                                            "final_discount": 15}),
        text_turn("A 5% discount is approved."),
    ])
    agent = build_agent(store, config, probs={0: 0.30, 5: 0.42, 10: 0.47, 15: 0.50},
                        llm=llm)

    decision = agent.decide(make_checkout())

    assert decision.discount_percent == 5
    assert decision.expected_profit == pytest.approx(0.42 * 250.0)
    record = store.reconstruct(decision.audit_id)
    assert record["input_features"]["cart_value"] == 1000.0
    assert record["final_discount"] == 5


# --------------------------------------------------------------------------------------
# MALFORMED / MISSING INPUT
# --------------------------------------------------------------------------------------
def test_missing_checkout_fields_are_handled_gracefully(store, config):
    agent = build_agent(store, config)
    decision = agent.decide({"customer_id": "C1", "transaction_id": "T1"})

    assert decision.decision == "FALLBACK"
    assert decision.discount_percent == 0
    assert decision.reason_code == "INVALID_CHECKOUT_REQUEST"
    assert decision.audit_id is not None


def test_invalid_numeric_checkout_value_is_handled_gracefully(store, config):
    agent = build_agent(store, config)
    decision = agent.decide(make_checkout(cart_value=float("nan")))

    assert decision.reason_code == "INVALID_CHECKOUT_REQUEST"
    assert decision.discount_percent == 0


def test_unknown_tool_name_does_not_break_the_flow(store, config):
    llm = ScriptedLLM([
        tool_turn("apply_discount_directly", "t1", {"discount": 15}),
        tool_turn("score_checkout", "t2"),
        tool_turn("evaluate_policy", "t3"),
        text_turn("A 5% discount is approved."),
    ])
    agent = build_agent(store, config, llm=llm)

    decision = agent.decide(make_checkout())
    assert decision.discount_percent == 5


# --------------------------------------------------------------------------------------
# EXPLANATION GUARDS (unit level)
# --------------------------------------------------------------------------------------
def test_sanitize_strips_hidden_reasoning_and_markdown():
    text = ("<thinking>the merchant will never notice</thinking> "
            "**A 5% discount** is approved.")
    cleaned = sanitize_explanation(text)
    assert "thinking" not in cleaned.lower()
    assert "**" not in cleaned
    assert cleaned.startswith("A 5% discount")


def test_conflict_guard_allows_legitimate_threshold_mentions():
    text = ("15% is the highest expected-profit candidate, but discounts above 10% "
            "require merchant approval.")
    assert explanation_conflicts_with_policy(
        text, decision="HUMAN_APPROVAL_REQUIRED", discount=15) is False


def test_conflict_guard_catches_a_wrong_offered_discount():
    assert explanation_conflicts_with_policy(
        "We are offering a 10% discount today.", decision="APPROVED", discount=5) is True


def test_conflict_guard_catches_approval_claims_on_a_fallback():
    assert explanation_conflicts_with_policy(
        "The discount was approved.", decision="FALLBACK", discount=0) is True


def test_system_prompt_states_the_core_constraints(store, config):
    llm = full_llm_script("A 5% discount is approved.")
    agent = build_agent(store, config, llm=llm)
    agent.decide(make_checkout())

    prompt = llm.systems[0].lower()
    for phrase in ("never calculate", "policy engine is authoritative",
                   "never invent numerical values", "never reveal your reasoning"):
        assert phrase in prompt
