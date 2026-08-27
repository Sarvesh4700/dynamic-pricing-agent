"""
api/main.py

Thin HTTP/web bridge over the EXISTING dynamic-pricing pipeline.

    browser  ->  FastAPI (this file)
                     -> src.checkout_pricing.price_checkout()
                            -> src.uplift_model  (T-learner)
                            -> src.policy_engine (approved discount)
                     -> src.checkout_pricing.create_razorpay_order_for_checkout()
                            -> src.razorpay_client.create_test_order()
                     -> Razorpay TEST MODE order_id
    browser  ->  Razorpay Standard Checkout (official checkout.js)
    browser  ->  FastAPI /api/verify-payment (server-side signature check)

THIS FILE CONTAINS NO PRICING LOGIC.
It does not compute discounts, uplift, expected profit or final prices, and it
never re-implements or overrides a policy-engine rule. Every number it returns
comes out of src/checkout_pricing.py. Its only jobs are: request validation,
refusing to trust client-supplied money fields, order bookkeeping, and
server-side Razorpay signature verification.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the project root importable so `src.*` resolves when uvicorn starts.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import razorpay
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

# The ONLY pricing entry points this API is allowed to use.
from src.checkout_pricing import (  # noqa: E402
    create_razorpay_order_for_checkout,
    price_checkout,
)

load_dotenv()

logger = logging.getLogger("dynamic_pricing.api")

FRONTEND_DIR = ROOT / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

# The policy engine can return decision == "HUMAN_APPROVAL_REQUIRED" (RULE 7).
# When True, this API honours that decision by refusing to auto-create a payable
# order. This is NOT a new business rule: it only acts on the decision string the
# policy engine already produced. Set to False if your demo should charge the
# pending-approval price anyway.
ENFORCE_HUMAN_APPROVAL_GATE = True

# Money/decision fields a browser must never be able to dictate. If a client
# sends these we drop them and report them back so the demo can show it happened.
UNTRUSTED_CLIENT_FIELDS = (
    "final_price",
    "approved_discount",
    "approved_discount_percent",
    "discount",
    "discount_percentage",
    "expected_profit",
    "pred_optimal_discount",
    "amount",
    "amount_paise",
    "policy",
    "decision",
    "payment_successful",
    "verified",
)

app = FastAPI(
    title="Dynamic Pricing Agent API",
    description="Web bridge for the existing T-learner + policy engine + Razorpay pipeline.",
    version="1.0.0",
)

if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ----------------------------------------------------------------------------------
# SERVER-SIDE ORDER BOOK
# ----------------------------------------------------------------------------------
# Razorpay's docs are explicit: when verifying a signature, use the order_id stored
# on YOUR server, not the one echoed back by the browser. This in-memory store is
# that record. It is process-local and cleared on restart -- fine for a demo, and
# deliberately not a new database.
_ORDER_STORE: Dict[str, Dict[str, Any]] = {}
_ORDER_STORE_LOCK = threading.Lock()


def _remember_order(order_id: str, record: Dict[str, Any]) -> None:
    with _ORDER_STORE_LOCK:
        _ORDER_STORE[order_id] = record


def _lookup_order(order_id: str) -> Optional[Dict[str, Any]]:
    with _ORDER_STORE_LOCK:
        record = _ORDER_STORE.get(order_id)
        return dict(record) if record else None


def _mark_order_paid(order_id: str, payment_id: str) -> None:
    with _ORDER_STORE_LOCK:
        if order_id in _ORDER_STORE:
            _ORDER_STORE[order_id]["status"] = "paid"
            _ORDER_STORE[order_id]["razorpay_payment_id"] = payment_id


# ----------------------------------------------------------------------------------
# REQUEST MODELS
# ----------------------------------------------------------------------------------
class CheckoutPayload(BaseModel):
    """Exactly the observable checkout/customer fields the existing pipeline needs.

    `extra="ignore"` is the security boundary: anything else the browser sends --
    including final_price or a discount it would like -- is discarded before the
    dict ever reaches price_checkout().
    """

    model_config = ConfigDict(extra="ignore")

    # identity
    customer_id: str = Field(min_length=1, max_length=64)
    transaction_id: Optional[str] = Field(default=None, max_length=40)

    # customer history (features + policy inputs)
    customer_type: str = Field(min_length=1, max_length=32)
    previous_purchases: int = Field(ge=0, le=10_000)
    previous_abandons: int = Field(ge=0, le=10_000)
    days_since_last_purchase: float = Field(ge=0, le=100_000)
    customer_lifetime_value: float = Field(ge=0, le=100_000_000)
    historical_discount_rate: float = Field(ge=0, le=100)
    recent_discount_count: int = Field(ge=0, le=1_000)
    days_since_last_discount: Optional[float] = Field(default=None, ge=0, le=100_000)

    # cart / session
    cart_value: float = Field(gt=0, le=10_000_000)
    items_count: int = Field(ge=1, le=1_000)
    category: str = Field(min_length=1, max_length=64)
    margin_percentage: float = Field(ge=0, le=100)
    device_type: str = Field(min_length=1, max_length=32)
    time_on_checkout_seconds: float = Field(ge=0, le=100_000)
    pages_viewed: int = Field(ge=0, le=10_000)
    payment_attempts: int = Field(ge=0, le=100)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)

    def to_checkout_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        if not data.get("transaction_id"):
            data["transaction_id"] = f"TXN_{uuid.uuid4().hex[:12].upper()}"
        return data


class PaymentVerificationPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    razorpay_payment_id: str = Field(min_length=1, max_length=128)
    razorpay_order_id: str = Field(min_length=1, max_length=128)
    razorpay_signature: str = Field(min_length=1, max_length=256)


# ----------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------
async def _ignored_client_fields(request: Request) -> List[str]:
    """Report which untrusted pricing fields the client tried to send."""
    try:
        body = await request.json()
    except Exception:
        return []
    if not isinstance(body, dict):
        return []
    return sorted(f for f in UNTRUSTED_CLIENT_FIELDS if f in body)


async def _run_pricing(checkout: Dict[str, Any]) -> Dict[str, Any]:
    """Call the existing pipeline off the event loop (it is sync and CPU-bound)."""
    try:
        return await run_in_threadpool(price_checkout, checkout)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Pricing pipeline unavailable: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid checkout for pricing: {exc}")


def _pricing_response(result: Dict[str, Any], ignored: List[str]) -> Dict[str, Any]:
    policy = result["policy"]
    cart_value = float(result["cart_value"])
    final_price = float(result["final_price"])
    return {
        "transaction_id": result.get("transaction_id"),
        "customer_id": result["customer_id"],
        "cart_value": cart_value,
        "approved_discount_percent": int(result["approved_discount_percent"]),
        "final_price": final_price,
        "savings": round(cart_value - final_price, 2),
        "policy": {
            "decision": policy["decision"],
            "reason_code": policy["reason_code"],
            "explanation": policy["explanation"],
            "model_recommended_discount": policy["model_recommended_discount"],
            "policy_selected_discount": policy["policy_selected_discount"],
            "expected_profit_selected": policy["expected_profit_selected"],
            "predicted_uplift_selected": policy["predicted_uplift_selected"],
            "policy_version": policy["policy_version"],
            "model_version": policy["model_version"],
            "policy_checks": policy["policy_checks"],
        },
        "model_predictions": result["model_predictions"],
        "ignored_client_supplied_fields": ignored,
    }


def _razorpay_key_id() -> str:
    key_id = os.getenv("RAZORPAY_KEY_ID")
    if not key_id:
        raise HTTPException(status_code=503, detail="Razorpay key id is not configured.")
    return key_id


def _verify_razorpay_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """Official SDK verification (HMAC-SHA256 of 'order_id|payment_id').

    This is verification only -- order CREATION still goes exclusively through
    src/razorpay_client.create_test_order(). The secret is read here and never
    returned, rendered or logged.
    """
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise HTTPException(status_code=503, detail="Razorpay credentials are not configured.")

    client = razorpay.Client(auth=(key_id, key_secret))
    try:
        outcome = client.utility.verify_payment_signature(
            {
                "razorpay_order_id": order_id,
                "razorpay_payment_id": payment_id,
                "razorpay_signature": signature,
            }
        )
    except razorpay.errors.SignatureVerificationError:
        return False
    # Some SDK versions return None on success, others return a boolean.
    return outcome is not False


# ----------------------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=500, detail="frontend/index.html is missing.")
    return FileResponse(str(INDEX_HTML))


@app.get("/api/health")
async def health():
    """Never reveals the secret -- only whether it is present."""
    return {
        "status": "ok",
        "razorpay_key_id_configured": bool(os.getenv("RAZORPAY_KEY_ID")),
        "razorpay_key_secret_configured": bool(os.getenv("RAZORPAY_KEY_SECRET")),
        "test_mode": (os.getenv("RAZORPAY_KEY_ID") or "").startswith("rzp_test_"),
    }


@app.post("/api/pricing")
async def api_pricing(payload: CheckoutPayload, request: Request):
    """Price a checkout. Read-only: no Razorpay order is created here."""
    ignored = await _ignored_client_fields(request)
    result = await _run_pricing(payload.to_checkout_dict())
    return _pricing_response(result, ignored)


@app.post("/api/create-order")
async def api_create_order(payload: CheckoutPayload, request: Request):
    """Price the checkout server-side, then create a Razorpay TEST MODE order for
    exactly that price. Any price/discount in the request body is discarded."""
    ignored = await _ignored_client_fields(request)
    checkout = payload.to_checkout_dict()

    # Pre-flight the pricing so we can honour a HUMAN_APPROVAL_REQUIRED decision
    # BEFORE money is involved. The pipeline is deterministic, so this produces
    # the same numbers as the order call below.
    quote = await _run_pricing(checkout)
    decision = quote["policy"]["decision"]

    if ENFORCE_HUMAN_APPROVAL_GATE and decision == "HUMAN_APPROVAL_REQUIRED":
        return JSONResponse(
            status_code=409,
            content={
                "order_created": False,
                "decision": decision,
                "reason_code": quote["policy"]["reason_code"],
                "explanation": quote["policy"]["explanation"],
                "approved_discount_percent": int(quote["approved_discount_percent"]),
                "final_price": float(quote["final_price"]),
                "detail": "The policy engine routed this discount to human approval; "
                          "no payable order was created.",
            },
        )

    try:
        result = await run_in_threadpool(create_razorpay_order_for_checkout, checkout)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Pricing pipeline unavailable: {exc}")
    except RuntimeError as exc:
        # Raised by src/razorpay_client.py for missing / non-test credentials.
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    order = result["razorpay_order"]
    final_price = float(result["final_price"])
    expected_paise = round(final_price * 100)

    # Hard invariant: the amount Razorpay will charge must equal the backend price.
    if int(order["amount"]) != expected_paise:
        logger.error("Order amount mismatch for order %s", order["id"])
        raise HTTPException(
            status_code=500,
            detail="Order amount does not match the backend-calculated final price.",
        )

    _remember_order(
        order["id"],
        {
            "order_id": order["id"],
            "amount_paise": int(order["amount"]),
            "currency": order["currency"],
            "final_price": final_price,
            "cart_value": float(result["cart_value"]),
            "approved_discount_percent": int(result["approved_discount_percent"]),
            "decision": result["policy"]["decision"],
            "reason_code": result["policy"]["reason_code"],
            "customer_id": result["customer_id"],
            "transaction_id": result.get("transaction_id"),
            "status": order.get("status", "created"),
        },
    )

    return {
        "order_created": True,
        "order_id": order["id"],
        "amount": int(order["amount"]),          # paise, authoritative
        "currency": order["currency"],
        "razorpay_key_id": _razorpay_key_id(),   # public key, required by checkout.js
        "cart_value": float(result["cart_value"]),
        "final_price": final_price,
        "approved_discount_percent": int(result["approved_discount_percent"]),
        "decision": result["policy"]["decision"],
        "reason_code": result["policy"]["reason_code"],
        "explanation": result["policy"]["explanation"],
        "transaction_id": result.get("transaction_id"),
        "ignored_client_supplied_fields": ignored,
    }


@app.post("/api/verify-payment")
async def api_verify_payment(payload: PaymentVerificationPayload):
    """Server-side signature verification. A browser cannot declare success."""
    record = _lookup_order(payload.razorpay_order_id)
    if record is None:
        return JSONResponse(
            status_code=400,
            content={
                "verified": False,
                "status": "unknown_order",
                "detail": "This order was not created by this server (or the server restarted).",
            },
        )

    verified = await run_in_threadpool(
        _verify_razorpay_signature,
        record["order_id"],                 # server-held order id, per Razorpay docs
        payload.razorpay_payment_id,
        payload.razorpay_signature,
    )

    if not verified:
        logger.warning("Signature verification FAILED for order %s", record["order_id"])
        return JSONResponse(
            status_code=400,
            content={
                "verified": False,
                "status": "signature_mismatch",
                "detail": "Payment signature verification failed. Order not fulfilled.",
            },
        )

    _mark_order_paid(record["order_id"], payload.razorpay_payment_id)
    logger.info("Signature verified for order %s", record["order_id"])

    return {
        "verified": True,
        "status": "payment_verified",
        "order_id": record["order_id"],
        "razorpay_payment_id": payload.razorpay_payment_id,
        "amount_paid": record["amount_paise"] / 100.0,
        "currency": record["currency"],
        "approved_discount_percent": record["approved_discount_percent"],
        "cart_value": record["cart_value"],
    }
