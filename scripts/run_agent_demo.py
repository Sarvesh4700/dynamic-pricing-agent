"""
scripts/run_agent_demo.py

Local, offline-capable demo of the Dynamic Pricing Agent's orchestration layer.

    python scripts/run_agent_demo.py
    python scripts/run_agent_demo.py --scenario high_discount
    SIMULATE_MODEL_FAILURE=true python scripts/run_agent_demo.py
    python scripts/run_agent_demo.py --simulate-model-failure timeout
    python scripts/run_agent_demo.py --no-llm

Runs with or without an ANTHROPIC_API_KEY: without one, the deterministic
ML + policy pipeline still produces and audits a decision.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

from src.agent import PricingAgent, render_merchant_summary  # noqa: E402
from src.agent_tools import CheckoutRequest, ModelScorer  # noqa: E402
from src.audit import DEFAULT_DB_PATH, AuditStore  # noqa: E402

DISCOUNT_LEVELS = [0, 5, 10, 15]


def base_checkout(**overrides) -> CheckoutRequest:
    payload = dict(
        customer_id="C104217",
        transaction_id="TXN_DEMO_0001",
        customer_type="returning",
        previous_purchases=3,
        previous_abandons=2,
        days_since_last_purchase=21,
        customer_lifetime_value=6400.0,
        historical_discount_rate=5.0,
        recent_discount_count=0,
        days_since_last_discount=34,
        cart_value=1850.0,
        items_count=3,
        category="fashion",
        margin_percentage=38.0,
        device_type="mobile",
        time_on_checkout_seconds=164.0,
        pages_viewed=5,
        payment_attempts=1,
        hour=21,
        day_of_week=5,
    )
    payload.update(overrides)
    return CheckoutRequest(**payload)


SCENARIOS = {
    "default": lambda: base_checkout(),
    "high_discount": lambda: base_checkout(
        transaction_id="TXN_DEMO_0002", customer_type="new", previous_purchases=0,
        previous_abandons=4, days_since_last_purchase=-1, customer_lifetime_value=0.0,
        cart_value=980.0, category="beauty", margin_percentage=52.0),
    "thin_margin": lambda: base_checkout(
        transaction_id="TXN_DEMO_0003", category="electronics", cart_value=41500.0,
        items_count=1, margin_percentage=11.0),
    "frequency_limited": lambda: base_checkout(
        transaction_id="TXN_DEMO_0004", recent_discount_count=2, days_since_last_discount=2),
    "invalid": lambda: {"customer_id": "C104217", "transaction_id": "TXN_DEMO_0005"},
}


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_model_output(model_output):
    print_header("1. MODEL PREDICTIONS (T-learner, loaded from models/uplift/)")
    if not model_output:
        print("  No model output - scoring failed or was unavailable for this checkout.")
        return
    print(f"  {'discount':>9} | {'P(convert)':>11} | {'uplift vs 0%':>13} | {'exp. profit':>12}")
    print("  " + "-" * 56)
    for d in DISCOUNT_LEVELS:
        uplift = model_output.get(f"pred_uplift_{d}") if d else 0.0
        print(f"  {str(d) + '%':>9} | {model_output[f'pred_p{d}']:>11.4f} | "
              f"{uplift:>13.4f} | {model_output[f'pred_expected_profit_{d}']:>12.2f}")
    print(f"\n  model's profit-maximizing discount: "
          f"{model_output['pred_optimal_discount']}% "
          f"(expected profit {model_output['pred_max_expected_profit']:.2f})")
    print(f"  model_version: {model_output.get('model_version')}")


def print_policy_output(policy_checks, decision):
    print_header("2. POLICY DECISION (deterministic - the source of truth)")
    print(f"  decision:                 {decision.decision}")
    print(f"  policy_selected_discount: {decision.discount_percent}%")
    print(f"  reason_code:              {decision.reason_code}")
    print(f"  model_recommended:        {decision.model_recommended_discount}")
    print(f"  policy_version:           {decision.policy_version}")
    print("\n  Rule checks:")
    for check in policy_checks:
        mark = "PASS" if check.get("passed") else "FAIL"
        scope = f"{check['discount']}%" if check.get("discount") is not None else "-"
        print(f"    [{mark}] {check['check']:<22} {scope:<5} {check.get('detail', '')}")


def main():
    parser = argparse.ArgumentParser(description="Dynamic Pricing Agent - agent layer demo")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="default")
    parser.add_argument("--simulate-model-failure", nargs="?", const="error",
                        choices=["error", "timeout"], default=None,
                        help="force a scoring failure without touching the environment")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the LLM entirely and run the deterministic pipeline")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    env_sim = os.getenv("SIMULATE_MODEL_FAILURE")
    simulate = args.simulate_model_failure
    print_header("DYNAMIC PRICING AGENT - AGENT / ORCHESTRATION LAYER DEMO")
    print(f"  scenario:                 {args.scenario}")
    print(f"  audit db:                 {args.db}")
    print(f"  SIMULATE_MODEL_FAILURE:   {env_sim or '(unset)'}")
    print(f"  --simulate-model-failure: {simulate or '(unset)'}")
    print(f"  ANTHROPIC_API_KEY:        "
          f"{'set (value never logged)' if os.getenv('ANTHROPIC_API_KEY') else 'not set'}")
    print(f"  AGENT_MODEL:              {os.getenv('AGENT_MODEL') or '(default)'}")

    if not os.path.exists(os.path.join("models", "uplift", "model_discount_0.joblib")):
        print("\n  NOTE: models/uplift/ artifacts not found. Generate them with:")
        print("        python src/data_generation.py && python src/uplift_model.py")
        print("        The demo continues - you will see the MODEL_UNAVAILABLE fallback.")

    store = AuditStore(args.db)
    agent = PricingAgent(
        scorer=ModelScorer(simulate_failure=simulate),
        audit_store=store,
        use_llm=not args.no_llm,
    )
    print(f"  agent model:              {agent.agent_model}")
    if agent.llm is None:
        print("  LLM:                      unavailable/disabled -> deterministic mode "
              "(checkout still decided)")

    checkout = SCENARIOS[args.scenario]()
    decision = agent.decide(checkout)

    # The agent exposes the raw tool outputs it used through the audit trail,
    # which is exactly what a reviewer would inspect.
    record = store.reconstruct(decision.audit_id) if decision.audit_id else None
    print_model_output((record or {}).get("model_output"))
    print_policy_output(decision.policy_checks, decision)

    print_header("3. AGENT RESPONSE (merchant-facing)")
    print(render_merchant_summary(decision))
    print(f"\n  explanation source:       {decision.explanation_source}")
    print(f"  llm_used:                 {decision.llm_used}")
    print(f"  degraded:                 {decision.degraded}")
    print(f"  error_type:               {decision.error_type or '(none)'}")

    print_header("4. STRUCTURED AGENT OUTPUT")
    print(json.dumps(decision.to_dict(core_only=True), indent=2))

    print_header("5. AUDIT TRAIL")
    if decision.audit_id is None:
        print("  WARNING: no audit record was written.")
    else:
        stored = store.get_decision(decision.audit_id)
        print(f"  audit_id:        {decision.audit_id}")
        print(f"  stored in:       {args.db}")
        print(f"  timestamp:       {stored['timestamp']}")
        print(f"  transaction_id:  {stored['transaction_id']}")
        print(f"  final_decision:  {stored['final_decision']} @ "
              f"{stored['final_discount']}% ({stored['reason_code']})")
        print(f"  agent_model:     {stored['agent_model']}")
        print(f"  success:         {bool(stored['success'])}   "
              f"error_type: {stored['error_type'] or '(none)'}")
        print(f"  reconstructable: input_features="
              f"{record.get('input_features') is not None}, "
              f"model_output={record.get('model_output') is not None}, "
              f"policy_output={record.get('policy_output') is not None}")
        print(f"  total rows in audit table: {store.count()}")

    print_header("DONE - checkout was never blocked by the agent layer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
