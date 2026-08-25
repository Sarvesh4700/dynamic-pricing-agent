import os
import razorpay
from dotenv import load_dotenv

load_dotenv()


def create_test_order(amount_rupees: float, receipt: str):
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")

    if not key_id or not key_secret:
        raise RuntimeError("Razorpay API credentials are not configured.")

    if not key_id.startswith("rzp_test_"):
        raise RuntimeError("Expected a Razorpay TEST MODE key.")

    amount_paise = round(amount_rupees * 100)

    if amount_paise < 1000:
        raise ValueError("Razorpay order amount must be at least ₹10.")

    client = razorpay.Client(auth=(key_id, key_secret))

    order = client.order.create(
        data={
            "amount": amount_paise,
            "currency": "INR",
            "receipt": receipt,
        }
    )

    return order