"""
tests/test_api.py

API-layer tests. No real Razorpay calls, no real payments, no model retraining.
The pricing pipeline is stubbed for the security tests so they run without the
joblib artefacts; one optional end-to-end test exercises the real pipeline and
skips if the trained models or config/policy.yaml are absent.
"""

import hashlib
import hmac
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_KEY_ID = "rzp_test_unittest0000"
TEST_KEY_SECRET = "unit_test_secret_not_real"

VALID_CHECKOUT = {
    "customer_id": "C104217",
    "transaction_id": "TXN_API_TEST_001",
    "customer_type": "returning",
    "previous_purchases": 3,
    "previous_abandons": 2,
    "days_since_last_purchase": 21,
    "customer_lifetime_value": 6400.0,
    "historical_discount_rate": 5.0,
    "recent_discount_count": 0,
    "days_since_last_discount": 34,
    "cart_value": 1850.0,
    "items_count": 3,
    "category": "fashion",
    "margin_percentage": 38.0,
    "device_type": "mobile",
    "time_on_checkout_seconds": 164.0,
    "pages_viewed": 5,
    "payment_attempts": 1,
    "hour": 21,
    "day_of_week": 5,
}


@pytest.fixture()
def main(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", TEST_KEY_ID)
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", TEST_KEY_SECRET)
    import api.main as api_main

    api_main._ORDER_STORE.clear()
    return api_main


@pytest.fixture()
def client(main):
    return TestClient(main.app)


# ----------------------------------------------------------------------------------
# Stubs for the existing pipeline (never re-implements policy logic; it only
# returns a fixed, known-good result so the API layer can be tested in isolation).
# ----------------------------------------------------------------------------------
BACKEND_DISCOUNT = 5
CALLS = {"checkouts": []}


def _stub_result(checkout, discount=BACKEND_DISCOUNT, decision="APPROVED",
                 reason="APPROVED"):
    cart = float(checkout["cart_value"])
    final = round(cart * (1 - discount / 100.0), 2)
    return {
        "transaction_id": checkout.get("transaction_id"),
        "customer_id": checkout["customer_id"],
        "cart_value": cart,
        "model_predictions": {
            "pred_p0": 0.30, "pred_p5": 0.36, "pred_p10": 0.38, "pred_p15": 0.39,
            "pred_optimal_discount": discount, "pred_max_expected_profit": 500.0,
        },
        "policy": {
            "decision": decision,
            "reason_code": reason,
            "explanation": "stubbed policy explanation",
            "model_recommended_discount": discount,
            "policy_selected_discount": discount,
            "expected_profit_selected": 500.0,
            "predicted_uplift_selected": 0.06,
            "policy_version": "1.0.0",
            "model_version": "t_learner_gbc_v1",
            "policy_checks": [],
        },
        "approved_discount_percent": discount,
        "final_price": final,
    }


def install_stubs(main, monkeypatch, discount=BACKEND_DISCOUNT,
                  decision="APPROVED", reason="APPROVED"):
    CALLS["checkouts"] = []

    def fake_price_checkout(checkout):
        CALLS["checkouts"].append(dict(checkout))
        return _stub_result(checkout, discount, decision, reason)

    def fake_create_order(checkout):
        CALLS["checkouts"].append(dict(checkout))
        result = _stub_result(checkout, discount, decision, reason)
        result["razorpay_order"] = {
            "id": "order_TESTFAKE0001",
            "amount": round(result["final_price"] * 100),
            "currency": "INR",
            "status": "created",
        }
        return result

    monkeypatch.setattr(main, "price_checkout", fake_price_checkout)
    monkeypatch.setattr(main, "create_razorpay_order_for_checkout", fake_create_order)


# ----------------------------------------------------------------------------------
# 1. GET /
# ----------------------------------------------------------------------------------
def test_root_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Dynamic Pricing" in response.text
    assert "checkout.razorpay.com/v1/checkout.js" in response.text
    assert TEST_KEY_SECRET not in response.text


def test_static_app_js_contains_no_secret(client):
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "KEY_SECRET" not in response.text
    assert TEST_KEY_SECRET not in response.text


# ----------------------------------------------------------------------------------
# 2. POST /api/pricing
# ----------------------------------------------------------------------------------
def test_pricing_returns_decision(main, client, monkeypatch):
    install_stubs(main, monkeypatch)
    response = client.post("/api/pricing", json=VALID_CHECKOUT)
    assert response.status_code == 200
    body = response.json()
    assert body["cart_value"] == 1850.0
    assert body["approved_discount_percent"] == BACKEND_DISCOUNT
    assert body["final_price"] == 1757.5
    assert body["policy"]["decision"] == "APPROVED"
    assert body["policy"]["reason_code"] in {
        "APPROVED", "NO_DISCOUNT_OPTIMAL", "INSUFFICIENT_UPLIFT",
        "PROFIT_FLOOR_VIOLATION", "DISCOUNT_FREQUENCY_LIMIT",
        "DISCOUNT_CAP_EXCEEDED", "MODEL_UNAVAILABLE",
    }


# ----------------------------------------------------------------------------------
# 3. POST /api/create-order uses the backend price
# ----------------------------------------------------------------------------------
def test_create_order_amount_matches_backend_price(main, client, monkeypatch):
    install_stubs(main, monkeypatch)
    response = client.post("/api/create-order", json=VALID_CHECKOUT)
    assert response.status_code == 200
    body = response.json()
    assert body["order_id"] == "order_TESTFAKE0001"
    assert body["final_price"] == 1757.5
    assert body["amount"] == 175750           # paise == final price
    assert body["currency"] == "INR"
    assert body["razorpay_key_id"] == TEST_KEY_ID
    assert "key_secret" not in response.text.lower()
    assert body["order_id"] in main._ORDER_STORE


# ----------------------------------------------------------------------------------
# 4 & 5. Client cannot override price or discount
# ----------------------------------------------------------------------------------
def test_client_final_price_is_ignored(main, client, monkeypatch):
    install_stubs(main, monkeypatch)
    tampered = dict(VALID_CHECKOUT, final_price=1, amount=100, expected_profit=999999)
    response = client.post("/api/create-order", json=tampered)
    assert response.status_code == 200
    body = response.json()
    assert body["final_price"] == 1757.5
    assert body["amount"] == 175750
    assert "final_price" in body["ignored_client_supplied_fields"]
    for checkout in CALLS["checkouts"]:
        assert "final_price" not in checkout
        assert "expected_profit" not in checkout


def test_client_discount_cannot_override_policy(main, client, monkeypatch):
    install_stubs(main, monkeypatch, discount=5)
    tampered = dict(VALID_CHECKOUT, approved_discount_percent=15, discount_percentage=15)
    response = client.post("/api/pricing", json=tampered)
    assert response.status_code == 200
    body = response.json()
    assert body["approved_discount_percent"] == 5
    assert body["final_price"] == 1757.5
    assert "approved_discount_percent" in body["ignored_client_supplied_fields"]
    for checkout in CALLS["checkouts"]:
        assert "approved_discount_percent" not in checkout
        assert "discount_percentage" not in checkout


def test_human_approval_decision_blocks_order(main, client, monkeypatch):
    install_stubs(main, monkeypatch, discount=15,
                  decision="HUMAN_APPROVAL_REQUIRED",
                  reason="HIGH_DISCOUNT_REQUIRES_APPROVAL")
    response = client.post("/api/create-order", json=VALID_CHECKOUT)
    assert response.status_code == 409
    assert response.json()["order_created"] is False


# ----------------------------------------------------------------------------------
# 6. Invalid input is rejected cleanly
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "override",
    [
        {"cart_value": -5},
        {"cart_value": "free"},
        {"hour": 99},
        {"day_of_week": 12},
        {"margin_percentage": 150},
        {"previous_purchases": -1},
    ],
)
def test_invalid_checkout_rejected(main, client, monkeypatch, override):
    install_stubs(main, monkeypatch)
    response = client.post("/api/pricing", json=dict(VALID_CHECKOUT, **override))
    assert response.status_code == 422
    assert CALLS["checkouts"] == []      # pipeline never invoked


def test_missing_required_field_rejected(main, client, monkeypatch):
    install_stubs(main, monkeypatch)
    payload = dict(VALID_CHECKOUT)
    payload.pop("cart_value")
    response = client.post("/api/pricing", json=payload)
    assert response.status_code == 422


# ----------------------------------------------------------------------------------
# 7. Signature verification
# ----------------------------------------------------------------------------------
def _signature(order_id, payment_id, secret=TEST_KEY_SECRET):
    return hmac.new(
        secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256
    ).hexdigest()


def test_invalid_signature_rejected(main, client, monkeypatch):
    install_stubs(main, monkeypatch)
    client.post("/api/create-order", json=VALID_CHECKOUT)

    response = client.post(
        "/api/verify-payment",
        json={
            "razorpay_order_id": "order_TESTFAKE0001",
            "razorpay_payment_id": "pay_TESTFAKE0001",
            "razorpay_signature": "0" * 64,
        },
    )
    assert response.status_code == 400
    assert response.json()["verified"] is False


def test_valid_signature_accepted(main, client, monkeypatch):
    install_stubs(main, monkeypatch)
    client.post("/api/create-order", json=VALID_CHECKOUT)

    order_id, payment_id = "order_TESTFAKE0001", "pay_TESTFAKE0001"
    response = client.post(
        "/api/verify-payment",
        json={
            "razorpay_order_id": order_id,
            "razorpay_payment_id": payment_id,
            "razorpay_signature": _signature(order_id, payment_id),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is True
    assert body["amount_paid"] == 1757.5


def test_unknown_order_rejected(main, client):
    response = client.post(
        "/api/verify-payment",
        json={
            "razorpay_order_id": "order_NEVER_CREATED",
            "razorpay_payment_id": "pay_x",
            "razorpay_signature": "abc",
        },
    )
    assert response.status_code == 400
    assert response.json()["verified"] is False


def test_frontend_boolean_cannot_declare_success(main, client, monkeypatch):
    install_stubs(main, monkeypatch)
    client.post("/api/create-order", json=VALID_CHECKOUT)
    response = client.post(
        "/api/verify-payment",
        json={
            "razorpay_order_id": "order_TESTFAKE0001",
            "razorpay_payment_id": "pay_TESTFAKE0001",
            "razorpay_signature": "bogus",
            "payment_successful": True,
            "verified": True,
        },
    )
    assert response.status_code == 400
    assert response.json()["verified"] is False


# ----------------------------------------------------------------------------------
# Optional end-to-end pricing test against the REAL models (no Razorpay call)
# ----------------------------------------------------------------------------------
@pytest.mark.skipif(
    not (ROOT / "models" / "uplift" / "model_discount_0.joblib").exists()
    or not (ROOT / "config" / "policy.yaml").exists(),
    reason="trained models and/or config/policy.yaml not present",
)
def test_pricing_with_real_pipeline(client):
    response = client.post("/api/pricing", json=VALID_CHECKOUT)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["approved_discount_percent"] <= 15
    assert body["final_price"] <= body["cart_value"]
    expected = round(1850.0 * (1 - body["approved_discount_percent"] / 100.0), 2)
    assert abs(body["final_price"] - expected) < 0.01
