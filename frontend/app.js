"use strict";

/*
 * Dynamic Pricing Demo
 *
 * Supports:
 *   1. Dataset-row testing from a locally selected CSV
 *   2. Manual customer/check-out testing
 *   3. Existing server-side pricing API
 *   4. Existing Razorpay Test Mode flow
 *
 * IMPORTANT:
 * The browser never decides the discount or final price.
 * Those values always come from the backend.
 */


// -----------------------------------------------------------------------------
// DOM HELPERS
// -----------------------------------------------------------------------------

function $(id) {
    return document.getElementById(id);
}


// -----------------------------------------------------------------------------
// FORM FIELD DEFINITIONS
// -----------------------------------------------------------------------------

const FIELD_DEFINITIONS = {
    customer_id: "string",
    customer_type: "string",

    previous_purchases: "number",
    previous_abandons: "number",
    days_since_last_purchase: "number",

    customer_lifetime_value: "number",
    historical_discount_rate: "number",
    recent_discount_count: "number",
    days_since_last_discount: "number",

    cart_value: "number",
    items_count: "number",
    category: "string",
    margin_percentage: "number",

    device_type: "string",
    time_on_checkout_seconds: "number",
    pages_viewed: "number",
    payment_attempts: "number",
    hour: "number",
    day_of_week: "number"
};


// -----------------------------------------------------------------------------
// STATE
// -----------------------------------------------------------------------------

let datasetRows = [];
let latestPricingResult = null;
let latestOrder = null;


// -----------------------------------------------------------------------------
// INITIAL DEMO DATA
// -----------------------------------------------------------------------------

const DEFAULT_VALUES = {
    customer_id: "C104217",
    customer_type: "returning",

    previous_purchases: 3,
    previous_abandons: 2,
    days_since_last_purchase: 21,

    customer_lifetime_value: 6400,
    historical_discount_rate: 5,
    recent_discount_count: 0,
    days_since_last_discount: 34,

    cart_value: 1850,
    items_count: 3,
    category: "fashion",
    margin_percentage: 38,

    device_type: "mobile",
    time_on_checkout_seconds: 164,
    pages_viewed: 5,
    payment_attempts: 1,
    hour: 21,
    day_of_week: 5
};


// -----------------------------------------------------------------------------
// FORM VALUE HELPERS
// -----------------------------------------------------------------------------

function setField(id, value) {
    const element = $(id);

    if (!element) {
        return;
    }

    if (value === undefined || value === null || value === "") {
        return;
    }

    element.value = String(value);
}


function getField(id, type) {
    const element = $(id);

    if (!element) {
        throw new Error(`Missing form field: ${id}`);
    }

    const raw = element.value.trim();

    if (type === "string") {
        if (!raw) {
            throw new Error(`${id} cannot be empty.`);
        }

        return raw;
    }

    const value = Number(raw);

    if (!Number.isFinite(value)) {
        throw new Error(`${id} must be a valid number.`);
    }

    return value;
}


function populateForm(values) {
    for (const [field, type] of Object.entries(FIELD_DEFINITIONS)) {
        if (Object.prototype.hasOwnProperty.call(values, field)) {
            setField(field, values[field]);
        }
    }
}


function resetToDefaultValues() {
    populateForm(DEFAULT_VALUES);
}


// -----------------------------------------------------------------------------
// ERROR DISPLAY
// -----------------------------------------------------------------------------

function showError(message) {
    const box = $("error-box");

    box.textContent = message;
    box.classList.add("visible");
}


function clearError() {
    const box = $("error-box");

    box.textContent = "";
    box.classList.remove("visible");
}


// -----------------------------------------------------------------------------
// MODE SWITCHING
// -----------------------------------------------------------------------------

function updateInputMode() {
    const mode = document.querySelector(
        'input[name="input-mode"]:checked'
    ).value;

    const datasetControls = $("dataset-controls");

    if (mode === "dataset") {
        datasetControls.classList.add("visible");
    } else {
        datasetControls.classList.remove("visible");
    }
}


// -----------------------------------------------------------------------------
// CSV PARSER
// -----------------------------------------------------------------------------

/*
 * Small CSV parser supporting:
 *   - commas
 *   - quoted fields
 *   - escaped quotes
 *   - Windows CRLF line endings
 *
 * This keeps dataset loading entirely in the browser.
 */

function parseCSV(text) {
    const rows = [];

    let row = [];
    let value = "";
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        const next = text[i + 1];

        if (char === '"') {
            if (inQuotes && next === '"') {
                value += '"';
                i++;
            } else {
                inQuotes = !inQuotes;
            }

            continue;
        }

        if (char === "," && !inQuotes) {
            row.push(value);
            value = "";
            continue;
        }

        if ((char === "\n" || char === "\r") && !inQuotes) {
            if (char === "\r" && next === "\n") {
                i++;
            }

            row.push(value);
            value = "";

            if (row.some(cell => cell.trim() !== "")) {
                rows.push(row);
            }

            row = [];
            continue;
        }

        value += char;
    }

    row.push(value);

    if (row.some(cell => cell.trim() !== "")) {
        rows.push(row);
    }

    if (rows.length < 2) {
        throw new Error("CSV does not contain any data rows.");
    }

    const headers = rows[0].map(header => header.trim());

    return rows.slice(1).map(cells => {
        const object = {};

        headers.forEach((header, index) => {
            object[header] = (cells[index] ?? "").trim();
        });

        return object;
    });
}


// -----------------------------------------------------------------------------
// DATASET VALIDATION
// -----------------------------------------------------------------------------

const REQUIRED_DATASET_FIELDS = [
    "customer_id",
    "customer_type",

    "previous_purchases",
    "previous_abandons",
    "days_since_last_purchase",

    "customer_lifetime_value",
    "historical_discount_rate",
    "recent_discount_count",
    "days_since_last_discount",

    "cart_value",
    "items_count",
    "category",
    "margin_percentage",

    "device_type",
    "time_on_checkout_seconds",
    "pages_viewed",
    "payment_attempts",
    "hour",
    "day_of_week"
];


function validateDatasetColumns(row) {
    const missing = REQUIRED_DATASET_FIELDS.filter(
        field => !(field in row)
    );

    if (missing.length > 0) {
        throw new Error(
            "CSV is missing required columns: " + missing.join(", ")
        );
    }
}


/*
 * The API currently validates the two "days since" fields as >= 0.
 *
 * The generated dataset uses -1 as a sentinel for "never happened".
 * Therefore we only load rows that are directly valid for the current API.
 *
 * This does NOT modify the source CSV.
 */
function rowIsUsableForCurrentAPI(row) {
    const lastPurchase = Number(row.days_since_last_purchase);
    const lastDiscount = Number(row.days_since_last_discount);

    if (!Number.isFinite(lastPurchase)) {
        return false;
    }

    if (!Number.isFinite(lastDiscount)) {
        return false;
    }

    if (lastPurchase < 0 || lastDiscount < 0) {
        return false;
    }

    return true;
}


// -----------------------------------------------------------------------------
// DATASET FILE LOADING
// -----------------------------------------------------------------------------

function handleDatasetFile(file) {
    if (!file) {
        return;
    }

    clearError();

    const reader = new FileReader();

    reader.onload = function(event) {
        try {
            const text = event.target.result;

            const parsedRows = parseCSV(text);

            if (parsedRows.length === 0) {
                throw new Error("No rows found in CSV.");
            }

            validateDatasetColumns(parsedRows[0]);

            const usableRows = parsedRows.filter(rowIsUsableForCurrentAPI);

            if (usableRows.length === 0) {
                throw new Error(
                    "No usable rows found. The current API requires " +
                    "non-negative days_since_last_purchase and " +
                    "days_since_last_discount."
                );
            }

            datasetRows = usableRows;

            $("file-status").textContent =
                `${file.name}: ${datasetRows.length.toLocaleString()} usable rows loaded`;

            $("dataset-row").max = datasetRows.length;
            $("dataset-row").value = 1;

            loadDatasetRow(1);

        } catch (error) {
            datasetRows = [];

            $("file-status").textContent =
                "Could not load dataset.";

            showError(error.message);
        }
    };

    reader.onerror = function() {
        showError("Could not read the selected CSV file.");
    };

    reader.readAsText(file);
}


// -----------------------------------------------------------------------------
// DATASET ROW -> FORM
// -----------------------------------------------------------------------------

function loadDatasetRow(rowNumber) {
    clearError();

    if (datasetRows.length === 0) {
        showError(
            "Load a transactions_train.csv or transactions_test.csv file first."
        );

        return;
    }

    const index = Number(rowNumber) - 1;

    if (!Number.isInteger(index) || index < 0 || index >= datasetRows.length) {
        showError(
            `Dataset row must be between 1 and ${datasetRows.length}.`
        );

        return;
    }

    const row = datasetRows[index];

    const values = {};

    for (const [field, type] of Object.entries(FIELD_DEFINITIONS)) {
        if (!(field in row)) {
            continue;
        }

        if (type === "number") {
            const numericValue = Number(row[field]);

            if (Number.isFinite(numericValue)) {
                values[field] = numericValue;
            }
        } else {
            values[field] = row[field];
        }
    }

    populateForm(values);

    $("dataset-row").value = rowNumber;

    $("file-status").textContent =
        `Loaded row ${rowNumber} → customer ${row.customer_id}`;
}


// -----------------------------------------------------------------------------
// COLLECT CHECKOUT PAYLOAD
// -----------------------------------------------------------------------------

function collectCheckoutPayload() {
    const payload = {};

    for (const [field, type] of Object.entries(FIELD_DEFINITIONS)) {
        payload[field] = getField(field, type);
    }

    /*
     * Do not send a client-selected discount.
     * The backend must calculate it.
     */

    return payload;
}


// -----------------------------------------------------------------------------
// SECURITY DEMO
// -----------------------------------------------------------------------------

function addSecurityDemoFields(payload) {
    if (!$("security-demo").checked) {
        return;
    }

    /*
     * These values are intentionally fake.
     *
     * The backend should ignore them and calculate the real price itself.
     */

    payload.final_price = 1;
    payload.approved_discount_percent = 15;
    payload.expected_profit = 999999;
    payload.discount_percentage = 15;
}


// -----------------------------------------------------------------------------
// PRICING REQUEST
// -----------------------------------------------------------------------------

async function calculatePrice() {
    clearError();

    $("calculate-button").disabled = true;
    $("calculate-button").textContent = "Calculating...";

    try {
        const payload = collectCheckoutPayload();

        addSecurityDemoFields(payload);

        const response = await fetch("/api/pricing", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
        });

        let body;

        try {
            body = await response.json();
        } catch {
            throw new Error(
                `Server returned HTTP ${response.status}.`
            );
        }

        if (!response.ok) {
            throw new Error(
                body.detail ||
                body.explanation ||
                `Pricing request failed with HTTP ${response.status}.`
            );
        }

        latestPricingResult = body;

        displayPricingResult(body);

    } catch (error) {
        latestPricingResult = null;

        showError(error.message);

    } finally {
        $("calculate-button").disabled = false;
        $("calculate-button").textContent = "Calculate My Price";
    }
}


// -----------------------------------------------------------------------------
// DISPLAY PRICING RESULT
// -----------------------------------------------------------------------------

function formatMoney(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return "₹" + number.toFixed(2);
}


function formatPercent(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return `${number}%`;
}


function displayPricingResult(body) {
    const policy = body.policy || {};

    const decision = policy.decision || "UNKNOWN";

    $("original-price").textContent =
        formatMoney(body.cart_value);

    $("discount-value").textContent =
        formatPercent(body.approved_discount_percent);

    $("savings-value").textContent =
        formatMoney(body.savings);

    $("final-price").textContent =
        formatMoney(body.final_price);

    $("policy-decision").textContent =
        decision;

    $("reason-code").textContent =
        policy.reason_code || "—";

    $("model-recommendation").textContent =
        formatPercent(policy.model_recommended_discount);

    $("expected-profit").textContent =
        policy.expected_profit_selected !== undefined
            ? formatMoney(policy.expected_profit_selected)
            : "—";

    $("predicted-uplift").textContent =
        policy.predicted_uplift_selected !== undefined
            ? Number(policy.predicted_uplift_selected).toFixed(4)
            : "—";

    $("explanation").textContent =
        policy.explanation || "";

    // Decision styling
    const decisionElement = $("policy-decision");

    decisionElement.classList.remove(
        "approved",
        "pending",
        "rejected"
    );

    if (decision === "APPROVED") {
        decisionElement.classList.add("approved");
    } else if (decision === "HUMAN_APPROVAL_REQUIRED") {
        decisionElement.classList.add("pending");
    } else {
        decisionElement.classList.add("rejected");
    }

    $("pricing-result").classList.add("visible");

    /*
     * Only allow payment when the policy actually approved the price.
     */

    const canPay =
        decision === "APPROVED";

    $("pay-button").disabled = !canPay;

    if (!canPay) {
        $("pay-button").textContent =
            decision === "HUMAN_APPROVAL_REQUIRED"
                ? "Human approval required"
                : "Payment unavailable";
    } else {
        $("pay-button").textContent =
            `Pay ${formatMoney(body.final_price)}`;
    }

    /*
     * If the security demo was enabled, show what the backend ignored.
     */

    if (
        Array.isArray(body.ignored_client_supplied_fields) &&
        body.ignored_client_supplied_fields.length > 0
    ) {
        const ignoredText =
            " Security test: backend ignored client fields: " +
            body.ignored_client_supplied_fields.join(", ");

        $("explanation").textContent += ignoredText;
    }
}


// -----------------------------------------------------------------------------
// CREATE RAZORPAY ORDER
// -----------------------------------------------------------------------------

async function createOrder() {
    clearError();

    if (!latestPricingResult) {
        showError("Calculate a price before attempting payment.");
        return;
    }

    const decision =
        latestPricingResult.policy?.decision;

    if (decision !== "APPROVED") {
        showError(
            "Payment cannot proceed because the policy decision is not APPROVED."
        );

        return;
    }

    $("pay-button").disabled = true;
    $("pay-button").textContent = "Creating order...";

    try {
        const payload = collectCheckoutPayload();

        /*
         * Deliberately do NOT send:
         *   final_price
         *   approved_discount_percent
         *   amount
         *   expected_profit
         *
         * The backend recalculates everything.
         */

        const response = await fetch("/api/create-order", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
        });

        let body;

        try {
            body = await response.json();
        } catch {
            throw new Error(
                `Server returned HTTP ${response.status}.`
            );
        }

        if (!response.ok) {
            throw new Error(
                body.detail ||
                body.explanation ||
                body.message ||
                `Order creation failed with HTTP ${response.status}.`
            );
        }

        if (!body.order_created) {
            throw new Error(
                body.detail ||
                "The backend did not create a payable order."
            );
        }

        latestOrder = body;

        openRazorpayCheckout(body);

    } catch (error) {
        showError(error.message);

        $("pay-button").disabled = false;
        $("pay-button").textContent =
            `Pay ${formatMoney(latestPricingResult.final_price)}`;
    }
}


// -----------------------------------------------------------------------------
// RAZORPAY CHECKOUT
// -----------------------------------------------------------------------------

function openRazorpayCheckout(order) {
    const options = {

        key: order.razorpay_key_id,

        amount: order.amount,

        currency: order.currency || "INR",

        name: "Dynamic Pricing Demo",

        description: "Personalized checkout price",

        order_id: order.order_id,

        prefill: {
            name: order.customer_id || "Test Customer"
        },

        theme: {
            color: "#2878c8"
        },

        handler: async function(response) {

            await verifyPayment(response);
        },

        modal: {
            ondismiss: function() {

                if (latestPricingResult) {
                    $("pay-button").disabled = false;

                    $("pay-button").textContent =
                        `Pay ${formatMoney(latestPricingResult.final_price)}`;
                }
            }
        }
    };

    const razorpay = new Razorpay(options);

    razorpay.on("payment.failed", function(response) {

        showPaymentResult(
            false,
            {
                verified: false,
                status: "payment_failed",
                detail:
                    response.error?.description ||
                    "Razorpay reported a payment failure."
            }
        );

        if (latestPricingResult) {
            $("pay-button").disabled = false;

            $("pay-button").textContent =
                `Pay ${formatMoney(latestPricingResult.final_price)}`;
        }
    });

    razorpay.open();
}


// -----------------------------------------------------------------------------
// SERVER-SIDE PAYMENT VERIFICATION
// -----------------------------------------------------------------------------

async function verifyPayment(razorpayResponse) {
    try {

        const response = await fetch("/api/verify-payment", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                razorpay_order_id:
                    razorpayResponse.razorpay_order_id,

                razorpay_payment_id:
                    razorpayResponse.razorpay_payment_id,

                razorpay_signature:
                    razorpayResponse.razorpay_signature
            })
        });

        let body;

        try {
            body = await response.json();
        } catch {
            throw new Error(
                `Payment verification returned HTTP ${response.status}.`
            );
        }

        if (!response.ok || body.verified !== true) {

            showPaymentResult(false, body);

            return;
        }

        showPaymentResult(true, body);

    } catch (error) {

        showPaymentResult(
            false,
            {
                verified: false,
                status: "verification_request_failed",
                detail: error.message
            }
        );
    }
}


// -----------------------------------------------------------------------------
// PAYMENT RESULT DISPLAY
// -----------------------------------------------------------------------------

function showPaymentResult(success, body) {
    const resultCard = $("payment-result");
    const message = $("payment-message");
    const json = $("payment-json");

    resultCard.classList.add("visible");

    if (success) {

        message.textContent =
            `Payment verified server-side. Paid ${formatMoney(body.amount_paid)}.`;

    } else {

        message.textContent =
            body.detail ||
            "Payment could not be verified.";
    }

    json.textContent =
        JSON.stringify(body, null, 2);

    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// -----------------------------------------------------------------------------
// EVENT LISTENERS
// -----------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", function() {

    // Mode switching
    document
        .querySelectorAll('input[name="input-mode"]')
        .forEach(radio => {
            radio.addEventListener("change", updateInputMode);
        });


    // Dataset file
    $("dataset-file").addEventListener(
        "change",
        function(event) {

            const file = event.target.files?.[0];

            handleDatasetFile(file);
        }
    );


    // Load selected dataset row
    $("load-row-button").addEventListener(
        "click",
        function() {

            loadDatasetRow(
                Number($("dataset-row").value)
            );
        }
    );


    // Calculate
    $("calculate-button").addEventListener(
        "click",
        calculatePrice
    );


    // Pay
    $("pay-button").addEventListener(
        "click",
        createOrder
    );


    // Initial state
    resetToDefaultValues();
    updateInputMode();

});