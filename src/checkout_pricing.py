import os
import sys
from pathlib import Path

import pandas as pd
import joblib
from dotenv import load_dotenv

# Allow this file to be run directly from the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.uplift_model import (
    DISCOUNT_LEVELS,
    FEATURE_COLS,
    predict_potential_outcomes,
    calculate_uplift,
    calculate_expected_profit,
)

from src.policy_engine import (
    CheckoutDecisionRequest,
    load_policy_config,
    evaluate_policy,
)

from src.razorpay_client import create_test_order


load_dotenv()


MODEL_DIR = ROOT / "models" / "uplift"


def load_models():
    """Load the four already-trained T-learner models."""
    models = {}

    for discount in DISCOUNT_LEVELS:
        path = MODEL_DIR / f"model_discount_{discount}.joblib"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing trained model: {path}"
            )

        models[discount] = joblib.load(path)

    return models


def calculate_final_price(cart_value: float, discount_percent: int) -> float:
    """Calculate the customer-facing price after the approved discount."""
    return round(
        cart_value * (1 - discount_percent / 100.0),
        2,
    )


def price_checkout(checkout: dict):
    """
    Run the complete deterministic pricing pipeline:

        checkout data
            ↓
        T-learner
            ↓
        uplift
            ↓
        expected profit
            ↓
        policy engine
            ↓
        approved discount
            ↓
        final price
    """

    # ------------------------------------------------------------------
    # 1. Load trained models
    # ------------------------------------------------------------------
    models = load_models()

    # The model expects a DataFrame with the same feature columns
    # used during training.
    model_input = pd.DataFrame([checkout])

    # Verify that all model features are present.
    missing = [col for col in FEATURE_COLS if col not in model_input.columns]

    if missing:
        raise ValueError(
            f"Checkout is missing model features: {missing}"
        )

    # ------------------------------------------------------------------
    # 2. Predict conversion probability under all four discounts
    # ------------------------------------------------------------------
    predictions = predict_potential_outcomes(
        models,
        model_input,
    )

    # ------------------------------------------------------------------
    # 3. Calculate uplift
    # ------------------------------------------------------------------
    predictions = calculate_uplift(predictions)

    # ------------------------------------------------------------------
    # 4. Calculate expected profit
    # ------------------------------------------------------------------
    predictions = calculate_expected_profit(predictions)

    row = predictions.iloc[0]

    # ------------------------------------------------------------------
    # 5. Build the request expected by the policy engine
    # ------------------------------------------------------------------
    policy_request = CheckoutDecisionRequest(
        # Customer information
        customer_id=checkout["customer_id"],
        customer_type=checkout["customer_type"],
        previous_purchases=checkout["previous_purchases"],
        previous_abandons=checkout["previous_abandons"],
        historical_discount_rate=checkout["historical_discount_rate"],
        recent_discount_count=checkout["recent_discount_count"],
        days_since_last_discount=checkout["days_since_last_discount"],

        # Checkout information
        cart_value=checkout["cart_value"],
        category=checkout["category"],
        margin_percentage=checkout["margin_percentage"],
        device_type=checkout["device_type"],
        time_on_checkout_seconds=checkout["time_on_checkout_seconds"],
        pages_viewed=checkout["pages_viewed"],
        payment_attempts=checkout["payment_attempts"],
        hour=checkout["hour"],
        day_of_week=checkout["day_of_week"],

        # ML outputs
        pred_p0=float(row["pred_p0"]),
        pred_p5=float(row["pred_p5"]),
        pred_p10=float(row["pred_p10"]),
        pred_p15=float(row["pred_p15"]),

        pred_expected_profit_0=float(
            row["pred_expected_profit_0"]
        ),
        pred_expected_profit_5=float(
            row["pred_expected_profit_5"]
        ),
        pred_expected_profit_10=float(
            row["pred_expected_profit_10"]
        ),
        pred_expected_profit_15=float(
            row["pred_expected_profit_15"]
        ),

        pred_optimal_discount=int(
            row["pred_optimal_discount"]
        ),
        pred_max_expected_profit=float(
            row["pred_max_expected_profit"]
        ),
    )

    # ------------------------------------------------------------------
    # 6. Let the EXISTING policy engine make the final decision
    # ------------------------------------------------------------------
    policy_config = load_policy_config()

    policy_result = evaluate_policy(
        policy_request,
        policy_config,
    )

    approved_discount = int(
        policy_result["policy_selected_discount"]
    )

    # ------------------------------------------------------------------
    # 7. Calculate actual customer-facing price
    # ------------------------------------------------------------------
    final_price = calculate_final_price(
        checkout["cart_value"],
        approved_discount,
    )

    result = {
        "transaction_id": checkout.get("transaction_id"),
        "customer_id": checkout["customer_id"],
        "cart_value": checkout["cart_value"],

        "model_predictions": {
            "pred_p0": float(row["pred_p0"]),
            "pred_p5": float(row["pred_p5"]),
            "pred_p10": float(row["pred_p10"]),
            "pred_p15": float(row["pred_p15"]),
            "pred_optimal_discount": int(
                row["pred_optimal_discount"]
            ),
            "pred_max_expected_profit": float(
                row["pred_max_expected_profit"]
            ),
        },

        "policy": policy_result,

        "approved_discount_percent": approved_discount,
        "final_price": final_price,
    }

    return result


def create_razorpay_order_for_checkout(checkout: dict):
    """
    Run pricing first, then create a Razorpay TEST MODE order
    using the policy-approved final price.
    """

    result = price_checkout(checkout)

    receipt = checkout.get(
        "transaction_id",
        "DYNAMIC_PRICING_TEST",
    )

    order = create_test_order(
        amount_rupees=result["final_price"],
        receipt=receipt,
    )

    result["razorpay_order"] = {
        "id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "status": order["status"],
    }

    return result


def main():
    """
    Test checkout using the same shape as the existing agent demo.
    """

    checkout = {
        "customer_id": "C104217",
        "transaction_id": "TXN_RAZORPAY_001",

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

    print("=" * 70)
    print("DYNAMIC PRICING → RAZORPAY TEST MODE")
    print("=" * 70)

    result = create_razorpay_order_for_checkout(checkout)

    print("\nMODEL")
    print("-" * 70)

    for discount in DISCOUNT_LEVELS:
        probability = result["model_predictions"][
            f"pred_p{discount}"
        ]

        print(
            f"{discount:>2}% discount → "
            f"P(convert) = {probability:.4f}"
        )

    print(
        "\nModel optimal discount:",
        f"{result['model_predictions']['pred_optimal_discount']}%"
    )

    print("\nPOLICY ENGINE")
    print("-" * 70)

    policy = result["policy"]

    print("Decision:", policy["decision"])
    print(
        "Approved discount:",
        f"{result['approved_discount_percent']}%"
    )
    print("Reason:", policy["reason_code"])

    print("\nPAYMENT")
    print("-" * 70)

    print(
        "Original cart:",
        f"₹{result['cart_value']:.2f}"
    )

    print(
        "Final price:",
        f"₹{result['final_price']:.2f}"
    )

    razorpay_order = result["razorpay_order"]

    print(
        "Razorpay order ID:",
        razorpay_order["id"]
    )

    print(
        "Razorpay amount:",
        razorpay_order["amount"],
        "paise"
    )

    print(
        "Razorpay status:",
        razorpay_order["status"]
    )

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()