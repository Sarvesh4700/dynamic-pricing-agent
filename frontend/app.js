/*
 * frontend/app.js
 *
 * Display + Razorpay Standard Checkout only.
 * This file computes NO price and NO discount: it renders what the backend
 * returned and hands the backend order_id to Razorpay's official checkout.js.
 * The Razorpay Key Secret never reaches the browser; the public Key ID arrives
 * with the create-order response.
 */

const form = document.getElementById("checkout-form");
const calculateBtn = document.getElementById("calculate-btn");
const payBtn = document.getElementById("pay-btn");
const priceCard = document.getElementById("price-card");
const resultCard = document.getElementById("result-card");
const resultText = document.getElementById("result-text");
const resultJson = document.getElementById("result-json");
const ignoredNote = document.getElementById("ignored-note");

let lastQuote = null;

const rupees = (n) => "\u20B9" + Number(n).toFixed(2);

function num(id) {
  return Number(document.getElementById(id).value);
}
function str(id) {
  return document.getElementById(id).value.trim();
}

function buildCheckoutPayload() {
  const payload = {
    customer_id: str("customer_id"),
    customer_type: str("customer_type"),
    previous_purchases: num("previous_purchases"),
    previous_abandons: num("previous_abandons"),
    days_since_last_purchase: num("days_since_last_purchase"),
    customer_lifetime_value: num("customer_lifetime_value"),
    historical_discount_rate: num("historical_discount_rate"),
    recent_discount_count: num("recent_discount_count"),
    days_since_last_discount: num("days_since_last_discount"),
    cart_value: num("cart_value"),
    items_count: num("items_count"),
    category: str("category"),
    margin_percentage: num("margin_percentage"),
    device_type: str("device_type"),
    time_on_checkout_seconds: num("time_on_checkout_seconds"),
    pages_viewed: num("pages_viewed"),
    payment_attempts: num("payment_attempts"),
    hour: num("hour"),
    day_of_week: num("day_of_week")
  };

  // Security demonstration: these are deliberately bogus and must be ignored.
  if (document.getElementById("tamper").checked) {
    payload.final_price = 1;
    payload.approved_discount_percent = 15;
    payload.expected_profit = 999999;
  }
  return payload;
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  let data;
  try {
    data = await response.json();
  } catch (err) {
    data = { detail: "Server returned a non-JSON response." };
  }
  return { ok: response.ok, status: response.status, data };
}

function describeError(data) {
  if (!data) return "Unknown error.";
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((e) => `${(e.loc || []).join(".")}: ${e.msg}`)
      .join(" | ");
  }
  return JSON.stringify(data);
}

function renderQuote(data) {
  document.getElementById("original-price").textContent = rupees(data.cart_value);
  document.getElementById("discount").textContent = data.approved_discount_percent + "%";
  document.getElementById("savings").textContent = rupees(data.savings);
  document.getElementById("final-price").textContent = rupees(data.final_price);
  document.getElementById("decision").textContent = data.policy.decision;
  document.getElementById("reason").textContent = data.policy.reason_code;
  document.getElementById("model-rec").textContent =
    data.policy.model_recommended_discount === null
      ? "unavailable"
      : data.policy.model_recommended_discount + "%";
  document.getElementById("expected-profit").textContent =
    data.policy.expected_profit_selected === null
      ? "n/a"
      : rupees(data.policy.expected_profit_selected);
  document.getElementById("explanation").textContent = data.policy.explanation;

  const ignored = data.ignored_client_supplied_fields || [];
  if (ignored.length) {
    ignoredNote.textContent =
      "Backend ignored these client-supplied fields: " + ignored.join(", ");
    ignoredNote.classList.remove("hidden");
  } else {
    ignoredNote.classList.add("hidden");
  }

  payBtn.textContent = "Pay " + rupees(data.final_price);
  priceCard.classList.remove("hidden");
}

function showResult(message, cssClass, payload) {
  resultText.textContent = message;
  resultText.className = cssClass;
  resultJson.textContent = payload ? JSON.stringify(payload, null, 2) : "";
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  calculateBtn.disabled = true;
  calculateBtn.textContent = "Calculating...";
  resultCard.classList.add("hidden");

  const { ok, data } = await postJSON("/api/pricing", buildCheckoutPayload());

  calculateBtn.disabled = false;
  calculateBtn.textContent = "Calculate My Price";

  if (!ok) {
    priceCard.classList.add("hidden");
    showResult("Pricing failed: " + describeError(data), "bad", data);
    return;
  }
  lastQuote = data;
  renderQuote(data);
});

payBtn.addEventListener("click", async () => {
  if (!lastQuote) return;
  payBtn.disabled = true;
  payBtn.textContent = "Creating order...";

  // The order is priced again server-side; nothing about the price travels
  // from this page to the order endpoint.
  const { ok, status, data } = await postJSON("/api/create-order", buildCheckoutPayload());

  payBtn.disabled = false;
  payBtn.textContent = "Pay " + rupees(lastQuote.final_price);

  if (!ok) {
    const prefix = status === 409 ? "Order blocked by policy: " : "Could not create order: ";
    showResult(prefix + describeError(data), status === 409 ? "warn" : "bad", data);
    return;
  }

  const options = {
    key: data.razorpay_key_id,              // public Key ID only
    amount: data.amount,                    // paise, from the backend
    currency: data.currency,
    name: "Dynamic Pricing Demo",
    description: "Wireless Headphones (test mode)",
    order_id: data.order_id,                // backend-generated Razorpay order
    handler: async function (response) {
      showResult("Verifying payment on the server...", "muted", null);
      const verification = await postJSON("/api/verify-payment", {
        razorpay_payment_id: response.razorpay_payment_id,
        razorpay_order_id: response.razorpay_order_id,
        razorpay_signature: response.razorpay_signature
      });
      if (verification.ok && verification.data.verified) {
        showResult(
          "Payment verified server-side. Paid " +
            rupees(verification.data.amount_paid) + ".",
          "ok",
          verification.data
        );
      } else {
        showResult(
          "Payment could NOT be verified: " + describeError(verification.data),
          "bad",
          verification.data
        );
      }
    },
    prefill: { name: "Test Customer", email: "test@example.com", contact: "+919999999999" },
    notes: { transaction_id: data.transaction_id || "", customer_id: lastQuote.customer_id },
    theme: { color: "#2b6cb0" },
    modal: {
      ondismiss: function () {
        showResult("Checkout closed before payment completed.", "warn", null);
      }
    }
  };

  const rzp = new Razorpay(options);
  rzp.on("payment.failed", function (response) {
    showResult(
      "Payment failed: " + (response.error && response.error.description),
      "bad",
      response.error
    );
  });
  rzp.open();
});
