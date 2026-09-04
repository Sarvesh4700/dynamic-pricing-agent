import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# PROJECT ROOT
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -----------------------------------------------------------------------------
# CONTINUOUS MODEL
# -----------------------------------------------------------------------------

from src.continuous_response_model import (
    ContinuousResponseModel,
)


# -----------------------------------------------------------------------------
# POLICY ENGINE
# -----------------------------------------------------------------------------

from src.policy_engine import (
    CheckoutDecisionRequest,
    load_policy_config,
    evaluate_policy,
)


# -----------------------------------------------------------------------------
# RAZORPAY
# -----------------------------------------------------------------------------

from src.razorpay_client import create_test_order


load_dotenv()


# -----------------------------------------------------------------------------
# MODEL PATH
# -----------------------------------------------------------------------------

MODEL_DIR = (
    ROOT
    / "models"
    / "uplift_continuous"
)

MODEL_PATH = (
    MODEL_DIR
    / "continuous_response_model.joblib"
)


# -----------------------------------------------------------------------------
# CONTINUOUS MODEL LOADER
# -----------------------------------------------------------------------------

def load_continuous_model():
    """
    Load the trained continuous response model.

    The model was trained as a single estimator:
        P(conversion | customer features, discount)
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing continuous response model: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# -----------------------------------------------------------------------------
# DISCOUNT GRID
# -----------------------------------------------------------------------------

def build_discount_grid(
    max_discount_percent: float
):
    """
    Build the set of discounts the pricing engine will actually evaluate.

    The model can handle arbitrary discounts, but the policy config determines
    the maximum discount that can be offered.

    Example:
        max_discount_percent = 15

    gives:
        0%, 1%, 2%, ..., 15%
    """

    max_discount = float(
        max_discount_percent
    )

    if max_discount < 0:
        raise ValueError(
            "max_discount_percent cannot be negative"
        )

    discounts = []

    current = 0.0

    while current <= max_discount + 1e-9:

        discounts.append(
            round(current, 2)
        )

        current += 1.0

    return discounts


# -----------------------------------------------------------------------------
# FINAL PRICE
# -----------------------------------------------------------------------------

def calculate_final_price(
    cart_value: float,
    discount_percent: float,
) -> float:
    """
    Calculate customer-facing price.
    """

    return round(
        float(cart_value)
        * (
            1.0
            - float(discount_percent) / 100.0
        ),
        2,
    )


# -----------------------------------------------------------------------------
# EXPECTED PROFIT
# -----------------------------------------------------------------------------

def calculate_expected_profit(
    cart_value: float,
    margin_percentage: float,
    discount_percent: float,
    conversion_probability: float,
) -> float:
    """
    Same economic definition used by the existing project.

    cost = V * (1 - margin)

    price_after_discount =
        V * (1 - discount)

    profit_if_converted =
        price_after_discount - cost

    expected_profit =
        P(convert) * profit_if_converted
    """

    V = float(cart_value)

    margin = (
        float(margin_percentage)
        / 100.0
    )

    discount = (
        float(discount_percent)
        / 100.0
    )

    cost = V * (
        1.0 - margin
    )

    price_after_discount = V * (
        1.0 - discount
    )

    profit_if_converted = (
        price_after_discount
        - cost
    )

    expected_profit = (
        float(conversion_probability)
        * profit_if_converted
    )

    return float(
        expected_profit
    )


# -----------------------------------------------------------------------------
# MODEL PREDICTIONS
# -----------------------------------------------------------------------------

def predict_continuous_discounts(
    model,
    checkout: dict,
    discounts: list,
):
    """
    Predict P(conversion) for every requested discount.

    Returns:

        {
            0.0: probability,
            1.0: probability,
            ...
            15.0: probability
        }
    """

    model_input = pd.DataFrame(
        [checkout]
    )

    probabilities = {}

    for discount in discounts:

        # The trained model exposes this public prediction method.
        probability = model.predict_conversion_probability(
            model_input,
            discount_percentage=discount,
        )

        # Convert numpy scalar / array output into float.
        if hasattr(
            probability,
            "item"
        ):
            probability = probability.item()

        elif hasattr(
            probability,
            "__len__"
        ):
            probability = probability[0]

        probability = float(
            probability
        )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Model returned invalid probability "
                f"{probability} for discount {discount}%"
            )

        probabilities[
            float(discount)
        ] = probability

    return probabilities


# -----------------------------------------------------------------------------
# COMPLETE PRICING PIPELINE
# -----------------------------------------------------------------------------

def price_checkout(
    checkout: dict
):
    """
    Run the complete deterministic continuous pricing pipeline.

        checkout data
              ↓
        continuous response model
              ↓
        P(conversion | discount)
              ↓
        expected profit
              ↓
        policy engine
              ↓
        approved discount
              ↓
        final price
    """

    # -------------------------------------------------------------------------
    # 1. Load policy
    # -------------------------------------------------------------------------

    policy_config = (
        load_policy_config()
    )

    # -------------------------------------------------------------------------
    # 2. Load continuous model
    # -------------------------------------------------------------------------

    model = (
        load_continuous_model()
    )

    # -------------------------------------------------------------------------
    # 3. Build policy-safe continuous grid
    # -------------------------------------------------------------------------

    discounts = build_discount_grid(
        policy_config.max_discount_percent
    )

    # -------------------------------------------------------------------------
    # 4. Verify model features
    # -------------------------------------------------------------------------

    model_input = pd.DataFrame(
        [checkout]
    )

    # ContinuousResponseModel exposes FEATURE_COLS.
    feature_cols = getattr(
        model,
        "feature_cols",
        None,
    )

    if feature_cols is None:
        feature_cols = getattr(
            model,
            "FEATURE_COLS",
            None,
        )

    if feature_cols is not None:

        missing = [
            col
            for col in feature_cols
            if col not in model_input.columns
        ]

        if missing:
            raise ValueError(
                f"Checkout is missing model features: "
                f"{missing}"
            )

    # -------------------------------------------------------------------------
    # 5. Predict conversion probability at every discount
    # -------------------------------------------------------------------------

    discount_predictions = (
        predict_continuous_discounts(
            model,
            checkout,
            discounts,
        )
    )

    # -------------------------------------------------------------------------
    # 6. Calculate expected profit at every discount
    # -------------------------------------------------------------------------

    expected_profit_predictions = {}

    for discount, probability in (
        discount_predictions.items()
    ):

        expected_profit_predictions[
            float(discount)
        ] = calculate_expected_profit(
            cart_value=checkout[
                "cart_value"
            ],
            margin_percentage=checkout[
                "margin_percentage"
            ],
            discount_percent=discount,
            conversion_probability=probability,
        )

    # -------------------------------------------------------------------------
    # 7. ML economic optimum
    # -------------------------------------------------------------------------

    model_optimal_discount = max(
        expected_profit_predictions,
        key=expected_profit_predictions.get,
    )

    model_max_expected_profit = (
        expected_profit_predictions[
            model_optimal_discount
        ]
    )

    # -------------------------------------------------------------------------
    # 8. Build policy request
    # -------------------------------------------------------------------------

    policy_request = CheckoutDecisionRequest(

        # Customer information
        customer_id=checkout[
            "customer_id"
        ],

        customer_type=checkout[
            "customer_type"
        ],

        previous_purchases=checkout[
            "previous_purchases"
        ],

        previous_abandons=checkout[
            "previous_abandons"
        ],

        historical_discount_rate=checkout[
            "historical_discount_rate"
        ],

        recent_discount_count=checkout[
            "recent_discount_count"
        ],

        days_since_last_discount=checkout[
            "days_since_last_discount"
        ],

        # Checkout information
        cart_value=checkout[
            "cart_value"
        ],

        category=checkout[
            "category"
        ],

        margin_percentage=checkout[
            "margin_percentage"
        ],

        device_type=checkout[
            "device_type"
        ],

        time_on_checkout_seconds=checkout[
            "time_on_checkout_seconds"
        ],

        pages_viewed=checkout[
            "pages_viewed"
        ],

        payment_attempts=checkout[
            "payment_attempts"
        ],

        hour=checkout[
            "hour"
        ],

        day_of_week=checkout[
            "day_of_week"
        ],

        # Continuous ML outputs
        discount_predictions=(
            discount_predictions
        ),

        expected_profit_predictions=(
            expected_profit_predictions
        ),

        pred_optimal_discount=(
            model_optimal_discount
        ),

        pred_max_expected_profit=(
            model_max_expected_profit
        ),
    )

    # -------------------------------------------------------------------------
    # 9. Policy engine
    # -------------------------------------------------------------------------

    policy_result = evaluate_policy(
        policy_request,
        policy_config,
    )

    approved_discount = float(
        policy_result[
            "policy_selected_discount"
        ]
    )

    # -------------------------------------------------------------------------
    # 10. Customer-facing price
    # -------------------------------------------------------------------------

    final_price = (
        calculate_final_price(
            checkout["cart_value"],
            approved_discount,
        )
    )

    # -------------------------------------------------------------------------
    # 11. Return result
    # -------------------------------------------------------------------------

    result = {

        "transaction_id":
            checkout.get(
                "transaction_id"
            ),

        "customer_id":
            checkout[
                "customer_id"
            ],

        "cart_value":
            checkout[
                "cart_value"
            ],

        "model_predictions": {

            "discount_predictions":
                {
                    str(k): float(v)
                    for k, v in
                    discount_predictions.items()
                },

            "expected_profit_predictions":
                {
                    str(k): float(v)
                    for k, v in
                    expected_profit_predictions.items()
                },

            "pred_optimal_discount":
                float(
                    model_optimal_discount
                ),

            "pred_max_expected_profit":
                float(
                    model_max_expected_profit
                ),
        },

        "policy":
            policy_result,

        "approved_discount_percent":
            approved_discount,

        "final_price":
            final_price,
    }

    return result


# -----------------------------------------------------------------------------
# RAZORPAY
# -----------------------------------------------------------------------------

def create_razorpay_order_for_checkout(
    checkout: dict
):
    """
    Run pricing first, then create a Razorpay TEST MODE order
    using the policy-approved final price.
    """

    result = price_checkout(
        checkout
    )

    receipt = checkout.get(
        "transaction_id",
        "DYNAMIC_PRICING_TEST",
    )

    order = create_test_order(
        amount_rupees=result[
            "final_price"
        ],
        receipt=receipt,
    )

    result[
        "razorpay_order"
    ] = {

        "id":
            order["id"],

        "amount":
            order["amount"],

        "currency":
            order["currency"],

        "status":
            order["status"],
    }

    return result


# -----------------------------------------------------------------------------
# DEMO
# -----------------------------------------------------------------------------

def main():

    checkout = {

        "customer_id":
            "C104217",

        "transaction_id":
            "TXN_RAZORPAY_001",

        "customer_type":
            "returning",

        "previous_purchases":
            3,

        "previous_abandons":
            2,

        "days_since_last_purchase":
            21,

        "customer_lifetime_value":
            6400.0,

        "historical_discount_rate":
            5.0,

        "recent_discount_count":
            0,

        "days_since_last_discount":
            34,

        "cart_value":
            1850.0,

        "items_count":
            3,

        "category":
            "fashion",

        "margin_percentage":
            38.0,

        "device_type":
            "mobile",

        "time_on_checkout_seconds":
            164.0,

        "pages_viewed":
            5,

        "payment_attempts":
            1,

        "hour":
            21,

        "day_of_week":
            5,
    }

    print("=" * 75)
    print(
        "CONTINUOUS DYNAMIC PRICING → "
        "RAZORPAY TEST MODE"
    )
    print("=" * 75)

    result = (
        create_razorpay_order_for_checkout(
            checkout
        )
    )

    print("\nMODEL")
    print("-" * 75)

    predictions = result[
        "model_predictions"
    ][
        "discount_predictions"
    ]

    profits = result[
        "model_predictions"
    ][
        "expected_profit_predictions"
    ]

    for discount in sorted(
        predictions,
        key=lambda x: float(x)
    ):

        probability = predictions[
            discount
        ]

        expected_profit = profits[
            discount
        ]

        print(
            f"{float(discount):>5.1f}% discount "
            f"→ P(convert) = "
            f"{probability:.4f} "
            f"| Expected profit = "
            f"₹{expected_profit:.2f}"
        )

    print(
        "\nModel optimal discount:",
        f"{result['model_predictions']['pred_optimal_discount']:.2f}%"
    )

    print(
        "Model max expected profit:",
        f"₹{result['model_predictions']['pred_max_expected_profit']:.2f}"
    )

    print("\nPOLICY ENGINE")
    print("-" * 75)

    policy = result[
        "policy"
    ]

    print(
        "Decision:",
        policy["decision"]
    )

    print(
        "Approved discount:",
        f"{result['approved_discount_percent']:.2f}%"
    )

    print(
        "Reason:",
        policy["reason_code"]
    )

    print(
        "Explanation:",
        policy["explanation"]
    )

    print("\nPAYMENT")
    print("-" * 75)

    print(
        "Original cart:",
        f"₹{result['cart_value']:.2f}"
    )

    print(
        "Final price:",
        f"₹{result['final_price']:.2f}"
    )

    razorpay_order = (
        result["razorpay_order"]
    )

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

    print("\n" + "=" * 75)
    print("SUCCESS")
    print("=" * 75)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()