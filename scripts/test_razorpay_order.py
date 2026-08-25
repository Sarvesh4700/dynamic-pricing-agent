from src.razorpay_client import create_test_order


order = create_test_order(
    amount_rupees=950,
    receipt="TEST_TXN_001",
)

print("ORDER CREATED SUCCESSFULLY")
print("Order ID:", order["id"])
print("Amount:", order["amount"], "paise")
print("Currency:", order["currency"])
print("Status:", order["status"])