"""
Tests for the Stage-2 continuous discount-response model. Covers the eight
required behaviours plus smoothness, heterogeneity, input validation, and the
profit convention.

Self-contained: if the trained artifact is absent, the fixture trains a small
model on a synthetic frame so the suite still runs. Tests that specifically
concern the saved artifact skip with an actionable message instead.

Run from the project root:
    python -m pytest tests/test_continuous_response_model.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# TREATMENT COORDINATE HELPERS
# =============================================================================

_LOG101 = np.log1p(100.0)


def _d_from_u(u):
    """Inverse of saturating_coordinate: u in [0,1] -> d in [0,100]."""
    return np.clip(
        np.expm1(np.asarray(u, dtype=float) * _LOG101),
        0.0,
        100.0,
    )


# =============================================================================
# SMOOTHNESS HELPERS
# =============================================================================

# The model is parameterised in:
#
#     u = log1p(d) / log1p(100)
#
# rather than directly in d. Therefore smoothness/curvature tests should use
# a uniform grid in u, which is the coordinate in which the spline basis lives.

CURVATURE_FLOOR = 1.0
PROBABILITY_TOL = 1e-3
RELATIVE_TOL = 0.02


def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


def _curve_u(model, rows, u):
    """Predict conversion probability on a grid supplied in native u."""
    return model.predict_response_curve(
        rows,
        _d_from_u(np.asarray(u, dtype=float)),
    )


def _d2_at(model, rows, u_star, h):
    """Central second-difference estimate at a fixed u_star."""
    c = _curve_u(
        model,
        rows,
        [u_star - h, u_star, u_star + h],
    )

    return (
        c[:, 0]
        - 2.0 * c[:, 1]
        + c[:, 2]
    ) / (h * h)


def _discount_spline_geometry(model):
    """
    Read the fitted discount-spline geometry directly from the model.

    Returns:
        {
            "u_lo": lower fitted spline boundary,
            "u_hi": upper fitted spline boundary,
            "interior": interior spline knots,
            "degree": spline degree,
        }

    Returns None if the installed sklearn version does not expose
    SplineTransformer.bsplines_.
    """
    feat = model.pipeline_.named_steps["features"]

    spline = getattr(feat, "discount_spline_", None)
    bsplines = getattr(spline, "bsplines_", None)

    if not bsplines:
        return None

    bs = bsplines[0]

    t = np.asarray(bs.t, dtype=float)
    k = int(bs.k)

    return {
        "u_lo": float(t[k]),
        "u_hi": float(t[t.size - k - 1]),
        "interior": np.unique(
            t[k + 1:t.size - k - 1]
        ).astype(float),
        "degree": k,
    }


def _max_turning_points_allowed(model):
    """
    Structural turning-point bound derived from the fitted spline.

    For fixed customer features, the logit is a cubic spline in u.
    """
    geo = _discount_spline_geometry(model)

    if geo is None:
        return 6

    n_interior = int(geo["interior"].size)

    flanks = (
        int(geo["u_lo"] > 1e-12)
        + int(geo["u_hi"] < 1.0 - 1e-12)
    )

    return n_interior + 2 + 2 * flanks


def _significant_turning_points(y, tol):
    """
    Count direction reversals whose retracement exceeds tol.

    This is a hysteresis / zig-zag style filter. Tiny numerical or
    economically irrelevant reversals are ignored.
    """
    y = np.asarray(y, dtype=float)

    if y.size < 3:
        return []

    turns = []

    direction = 0

    lo_v, lo_i = y[0], 0
    hi_v, hi_i = y[0], 0

    ext_v, ext_i = y[0], 0

    for i in range(1, y.size):
        v = float(y[i])

        if direction == 0:

            if v < lo_v:
                lo_v, lo_i = v, i

            if v > hi_v:
                hi_v, hi_i = v, i

            if hi_v - lo_v > tol:

                if hi_i > lo_i:
                    direction = 1
                    ext_v, ext_i = hi_v, hi_i
                else:
                    direction = -1
                    ext_v, ext_i = lo_v, lo_i

        elif direction == 1:

            if v > ext_v:
                ext_v, ext_i = v, i

            elif ext_v - v > tol:
                turns.append(ext_i)

                direction = -1
                ext_v, ext_i = v, i

        else:

            if v < ext_v:
                ext_v, ext_i = v, i

            elif v - ext_v > tol:
                turns.append(ext_i)

                direction = 1
                ext_v, ext_i = v, i

    return turns


# =============================================================================
# MODEL IMPORTS
# =============================================================================

from src.continuous_response_model import (
    ARBITRARY_DISCOUNTS,
    CATEGORICAL_FEATURES,
    DISCOUNT_COL,
    FORBIDDEN_COLUMNS,
    MODEL_PATH,
    NUMERICAL_FEATURES,
    REQUIRED_INFERENCE_COLS,
    TARGET_COL,
    ContinuousResponseModel,
    saturating_coordinate,
)


CATEGORIES = [
    "electronics",
    "fashion",
    "home",
    "beauty",
    "groceries",
    "accessories",
    "sports",
]


# =============================================================================
# SYNTHETIC DATA
# =============================================================================

def _synthetic_frame(n=4000, seed=0):
    """
    Small dataset with the real observable schema and a genuine, smooth,
    heterogeneous dependence on a continuous discount.
    """
    rng = np.random.default_rng(seed)

    prev_p = rng.poisson(1.2, n)
    prev_a = rng.poisson(1.4, n)

    d = np.clip(
        rng.beta(1.8, 4.5, n) * 100.0,
        0,
        100,
    )

    df = pd.DataFrame({
        "customer_type": rng.choice(
            ["new", "returning", "vip"],
            n,
        ),
        "category": rng.choice(
            CATEGORIES,
            n,
        ),
        "device_type": rng.choice(
            ["mobile", "desktop", "tablet"],
            n,
        ),
        "previous_purchases": prev_p,
        "previous_abandons": prev_a,
        "days_since_last_purchase": np.where(
            prev_p == 0,
            -1,
            rng.integers(0, 90, n),
        ),
        "customer_lifetime_value": np.round(
            prev_p * rng.uniform(200, 3000, n),
            2,
        ),
        "historical_discount_rate": np.round(
            rng.uniform(0, 60, n),
            2,
        ),
        "recent_discount_count": rng.integers(
            0,
            3,
            n,
        ),
        "days_since_last_discount": np.where(
            rng.random(n) < 0.3,
            -1,
            rng.integers(0, 60, n),
        ),
        "cart_value": np.round(
            rng.lognormal(7.2, 0.6, n),
            2,
        ),
        "items_count": rng.integers(
            1,
            12,
            n,
        ),
        "margin_percentage": np.round(
            rng.uniform(6, 55, n),
            2,
        ),
        "time_on_checkout_seconds": np.round(
            rng.lognormal(4.6, 0.5, n),
            1,
        ),
        "pages_viewed": rng.integers(
            1,
            10,
            n,
        ),
        "payment_attempts": rng.integers(
            1,
            4,
            n,
        ),
        "hour": rng.integers(
            0,
            24,
            n,
        ),
        "day_of_week": rng.integers(
            0,
            7,
            n,
        ),
        DISCOUNT_COL: np.round(
            d,
            4,
        ),
    })

    sensitivity = (
        0.25
        + 0.5 * (
            df["customer_type"] == "returning"
        )
    )

    logit = (
        -0.6
        + 0.35 * np.log1p(
            df["previous_purchases"]
        )
        - 0.40 * np.log1p(
            df["previous_abandons"]
        )
        - 0.30 * (
            np.log1p(df["cart_value"]) - 7.2
        )
        + 1.35
        * sensitivity
        * saturating_coordinate(
            df[DISCOUNT_COL]
        )
        + rng.normal(
            0,
            0.45,
            n,
        )
    )

    p = 1.0 / (
        1.0 + np.exp(-logit)
    )

    df[TARGET_COL] = (
        rng.random(n) < p
    ).astype(int)

    return df


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def model():
    if os.path.exists(MODEL_PATH):
        return ContinuousResponseModel.load(
            MODEL_PATH
        )

    m = ContinuousResponseModel(
        C=1.0,
        n_discount_knots=5,
        n_numeric_knots=4,
    )

    m.fit(
        _synthetic_frame()
    )

    return m


@pytest.fixture(scope="module")
def rows():
    return _synthetic_frame(
        n=40,
        seed=7,
    )[REQUIRED_INFERENCE_COLS]


@pytest.fixture(scope="module")
def smooth_rows(model, rows):
    """
    Rows used by the smoothness suite.

    Rows whose predictions hit the probability clipping guard are excluded
    because a deliberately clipped probability creates a flat region that
    would otherwise be interpreted as a smoothness defect.
    """
    sub = rows.head(12)

    c = _curve_u(
        model,
        sub,
        np.linspace(
            0.0,
            1.0,
            51,
        ),
    )

    ok = (
        (c.min(axis=1) > 1e-4)
        & (c.max(axis=1) < 1.0 - 1e-4)
    )

    kept = sub.iloc[
        np.flatnonzero(
            np.asarray(ok)
        )
    ]

    if len(kept) < 3:
        pytest.skip(
            "Too few unclipped rows to test curve smoothness."
        )

    return kept


# =============================================================================
# 1. LOADS SUCCESSFULLY
# =============================================================================

def test_saved_artifact_loads():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(
            f"No artifact at {MODEL_PATH}. "
            f"Train first: python src/continuous_response_model.py"
        )

    m = ContinuousResponseModel.load(
        MODEL_PATH
    )

    assert isinstance(
        m,
        ContinuousResponseModel,
    )

    assert m.fitted_

    assert m.pipeline_ is not None


def test_metadata_written_and_leak_free():
    meta_path = os.path.join(
        os.path.dirname(MODEL_PATH),
        "feature_metadata.json",
    )

    if not os.path.exists(meta_path):
        pytest.skip(
            "Train the model first to produce feature_metadata.json."
        )

    import json

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta[
        "treatment_is_continuous"
    ] is True

    assert meta[
        "treatment_range"
    ] == [0.0, 100.0]

    assert not (
        set(meta["feature_cols"])
        & set(meta["forbidden_columns"])
    )


def test_old_discrete_artifacts_untouched():
    """Stage-2 must not disturb models/uplift/."""
    old = "models/uplift"

    if not os.path.isdir(old):
        pytest.skip(
            "models/uplift/ not present in this checkout."
        )

    new_dir = os.path.dirname(
        MODEL_PATH
    )

    assert os.path.abspath(old) != os.path.abspath(
        new_dir
    )

    for d in (0, 5, 10, 15):

        expected = os.path.join(
            old,
            f"model_discount_{d}.joblib",
        )

        if os.path.exists(expected):
            assert os.path.getsize(
                expected
            ) > 0


# =============================================================================
# 2, 3, 4. ENDPOINTS AND ARBITRARY DECIMALS
# =============================================================================

def test_predict_at_zero_discount(model, rows):
    p = model.predict_conversion_probability(
        rows,
        0.0,
    )

    assert p.shape == (
        len(rows),
    )

    assert np.all(
        np.isfinite(p)
    )

    assert np.all(
        (p > 0)
        & (p < 1)
    )


def test_predict_at_hundred_discount(model, rows):
    p = model.predict_conversion_probability(
        rows,
        100.0,
    )

    assert p.shape == (
        len(rows),
    )

    assert np.all(
        np.isfinite(p)
    )

    assert np.all(
        (p > 0)
        & (p < 1)
    )


@pytest.mark.parametrize(
    "d",
    ARBITRARY_DISCOUNTS,
)
def test_predict_arbitrary_decimal_discounts(
    model,
    rows,
    d,
):
    p = model.predict_conversion_probability(
        rows,
        d,
    )

    assert p.shape == (
        len(rows),
    )

    assert np.all(
        np.isfinite(p)
    )

    assert np.all(
        (p > 0)
        & (p < 1)
    )


def test_predict_extra_odd_values(model, rows):
    for d in [
        0.0001,
        1.0 / 3.0,
        49.999,
        63.4,
        99.9999,
        100.0,
    ]:
        p = model.predict_conversion_probability(
            rows,
            d,
        )

        assert np.all(
            np.isfinite(p)
        )


# =============================================================================
# 5. VALID PROBABILITIES
# =============================================================================

def test_predictions_are_valid_probabilities_over_dense_grid(
    model,
    rows,
):
    grid = np.arange(
        0.0,
        100.5,
        0.5,
    )

    curves = model.predict_response_curve(
        rows,
        grid,
    )

    assert curves.shape == (
        len(rows),
        grid.size,
    )

    assert np.all(
        np.isfinite(curves)
    )

    assert curves.min() > 0.0
    assert curves.max() < 1.0


# =============================================================================
# 6. DIFFERENT DISCOUNTS -> DIFFERENT PREDICTIONS
# =============================================================================

def test_different_discounts_change_predictions(
    model,
    rows,
):
    p0 = model.predict_conversion_probability(
        rows,
        0.0,
    )

    p50 = model.predict_conversion_probability(
        rows,
        50.0,
    )

    assert not np.allclose(
        p0,
        p50,
    )

    assert np.mean(
        p50 > p0
    ) > 0.5, (
        "A 50% discount should help most customers."
    )


def test_tiny_discount_change_moves_prediction_slightly(
    model,
    rows,
):
    """
    Genuinely continuous: 12.47 must differ from 12.48,
    but only a little.
    """
    a = model.predict_conversion_probability(
        rows,
        12.47,
    )

    b = model.predict_conversion_probability(
        rows,
        12.48,
    )

    assert not np.array_equal(
        a,
        b,
    )

    assert np.max(
        np.abs(a - b)
    ) < 0.01


# =============================================================================
# 7. NO DISCRETE ARMS
# =============================================================================

def test_single_model_no_discrete_arms(
    model,
    rows,
):
    assert isinstance(
        model,
        ContinuousResponseModel,
    )

    assert hasattr(
        model,
        "pipeline_",
    )

    assert not any(
        hasattr(model, a)
        for a in (
            "models_",
            "arm_models_",
            "arms_",
        )
    )

    off_grid = model.predict_conversion_probability(
        rows,
        37.77,
    )

    assert np.all(
        np.isfinite(off_grid)
    )

    est = model.pipeline_.named_steps[
        "clf"
    ]

    assert hasattr(
        est,
        "coef_",
    )

    assert est.coef_.shape[0] == 1, (
        "Expected one estimator, not one per arm."
    )


def test_discount_is_a_real_model_input(
    model,
    rows,
):
    """
    The discount must enter through the design matrix,
    and its basis must respond continuously.
    """
    feat = model.pipeline_.named_steps[
        "features"
    ]

    names = list(
        feat.get_feature_names_out()
    )

    assert any(
        n.startswith("disc__")
        for n in names
    )

    assert any(
        "__x_u" in n
        for n in names
    ), "Missing heterogeneity interactions."

    u = saturating_coordinate(
        [0.0, 7.38, 50.0, 100.0]
    )

    assert np.all(
        np.diff(u) > 0
    )

    assert abs(
        u[0]
    ) < 1e-12

    assert abs(
        u[-1] - 1.0
    ) < 1e-12


# =============================================================================
# 8. NO GROUND TRUTH NEEDED AT INFERENCE
# =============================================================================

def test_no_ground_truth_columns_required(
    model,
    rows,
):
    assert not (
        set(REQUIRED_INFERENCE_COLS)
        & FORBIDDEN_COLUMNS
    )

    minimal = rows[
        REQUIRED_INFERENCE_COLS
    ].copy()

    assert not (
        set(minimal.columns)
        & FORBIDDEN_COLUMNS
    )

    p = model.predict_conversion_probability(
        minimal,
        23.91,
    )

    assert np.all(
        np.isfinite(p)
    )


def test_ground_truth_columns_are_rejected_if_smuggled_in(
    model,
    rows,
):
    poisoned = rows.copy()

    poisoned[
        "price_sensitivity"
    ] = 0.9

    poisoned[
        "true_prob_d0"
    ] = 0.5

    p_poisoned = model.predict_conversion_probability(
        poisoned,
        20.0,
    )

    p_clean = model.predict_conversion_probability(
        rows,
        20.0,
    )

    np.testing.assert_allclose(
        p_poisoned,
        p_clean,
        rtol=0,
        atol=0,
    )


# =============================================================================
# SHAPE / BEHAVIOUR
# =============================================================================

def test_second_derivative_estimate_converges_at_fixed_points(
    model,
    smooth_rows,
):
    """
    Correct C2 diagnostic: keep u_star fixed and refine h.

    For a C2 point:

        D2_h p(u_star) -> p''(u_star)

    as h -> 0.

    At a genuine slope discontinuity, the second-difference estimate
    behaves approximately like J/h and therefore keeps growing as h
    is reduced.

    This replaces the old max-over-grid refinement test, which can
    legitimately increase for a perfectly smooth function simply because
    finer grids resolve the location of maximum curvature more accurately.
    """
    geo = _discount_spline_geometry(
        model
    )

    probes = []

    if geo is not None:
        probes += [
            float(v)
            for v in np.r_[
                geo["u_lo"],
                geo["interior"],
                geo["u_hi"],
            ]
        ]

    probes += [
        float(v)
        for v in np.linspace(
            0.03,
            0.97,
            11,
        )
    ]

    probes = sorted({
        round(v, 12)
        for v in probes
        if 0.02 < v < 0.98
    })

    assert probes

    hs = [
        1e-2,
        5e-3,
        2.5e-3,
        1.25e-3,
    ]

    for u_star in probes:

        est = np.column_stack([
            _d2_at(
                model,
                smooth_rows,
                u_star,
                h,
            )
            for h in hs
        ])

        assert np.all(
            np.isfinite(est)
        ), (
            f"non-finite curvature at u={u_star}"
        )

        coarse = np.abs(
            est[:, 0]
        )

        fine = np.abs(
            est[:, -1]
        )

        allowed = (
            1.5
            * np.maximum(
                coarse,
                CURVATURE_FLOOR,
            )
        )

        bad = np.flatnonzero(
            fine > allowed
        )

        assert bad.size == 0, (
            f"|d2p/du2| keeps growing under refinement "
            f"at u={u_star:.6f}; estimates="
            f"{est[bad].tolist()}"
        )


def test_curvature_bounded_in_treatment_coordinate(
    model,
    rows,
):
    """
    Bound |d2p/du2| in the coordinate the model is actually built in.
    """
    u = np.linspace(
        0.0,
        1.0,
        401,
    )

    h_u = float(
        u[1] - u[0]
    )

    curves = model.predict_response_curve(
        rows,
        _d_from_u(u),
    )

    assert curves.shape == (
        len(rows),
        u.size,
    )

    assert np.all(
        np.isfinite(curves)
    )

    max_curvature = (
        float(
            np.max(
                np.abs(
                    np.diff(
                        curves,
                        n=2,
                        axis=1,
                    )
                )
            )
        )
        / (h_u ** 2)
    )

    assert max_curvature < 100.0, (
        f"|d2p/du2| = {max_curvature:.2f} "
        f"is implausibly large for a smooth response curve."
    )


def test_response_curve_is_reproduced_by_a_coarse_cubic_interpolant(
    model,
    smooth_rows,
):
    """
    Check that the response curve does not contain high-frequency wiggles.

    The treatment path is smooth and low-dimensional in the native u
    coordinate, so a 65-point cubic interpolation should closely reproduce
    the same curve on a grid four times finer.
    """
    scipy_interpolate = pytest.importorskip(
        "scipy.interpolate"
    )

    CubicSpline = scipy_interpolate.CubicSpline

    m = 65

    u_coarse = np.linspace(
        0.0,
        1.0,
        m,
    )

    u_fine = np.linspace(
        0.0,
        1.0,
        4 * (m - 1) + 1,
    )

    c_coarse = _curve_u(
        model,
        smooth_rows,
        u_coarse,
    )

    c_fine = _curve_u(
        model,
        smooth_rows,
        u_fine,
    )

    assert np.all(
        np.isfinite(c_coarse)
    )

    assert np.all(
        np.isfinite(c_fine)
    )

    for i in range(
        c_fine.shape[0]
    ):

        approx = CubicSpline(
            u_coarse,
            c_coarse[i],
        )(u_fine)

        err = np.abs(
            approx - c_fine[i]
        )

        amp = float(
            np.ptp(c_fine[i])
        )

        tol = max(
            PROBABILITY_TOL,
            RELATIVE_TOL * amp,
        )

        j = int(
            np.argmax(err)
        )

        assert err[j] <= tol, (
            f"row {i}: coarse cubic interpolation "
            f"error {err[j]:.3e} > tolerance "
            f"{tol:.3e}"
        )


def test_no_material_oscillation_in_response_curve(
    model,
    smooth_rows,
):
    """
    Ensure response curves do not contain material oscillations.

    A fixed 'turns <= 3' rule is not mathematically justified for this
    cubic-spline architecture. Instead, count only reversals whose
    probability retracement is large enough to matter.
    """
    u = np.linspace(
        0.0,
        1.0,
        601,
    )

    curves = _curve_u(
        model,
        smooth_rows,
        u,
    )

    max_turns = _max_turning_points_allowed(
        model
    )

    for i in range(
        curves.shape[0]
    ):

        y = curves[i]

        amp = float(
            np.ptp(y)
        )

        tol = max(
            PROBABILITY_TOL,
            RELATIVE_TOL * amp,
        )

        turns = _significant_turning_points(
            y,
            tol,
        )

        assert len(turns) <= max_turns, (
            f"row {i}: {len(turns)} material reversals "
            f"(allowed {max_turns})"
        )


def test_logit_is_exactly_piecewise_cubic_in_u(
    model,
    smooth_rows,
):
    """
    Architecture-conformance test.

    For a fixed customer, the treatment-dependent part of the logit is
    piecewise cubic in the native u coordinate. Therefore its fourth
    finite difference must be approximately zero inside each knot interval.
    """
    geo = _discount_spline_geometry(
        model
    )

    if geo is None:
        pytest.skip(
            "Spline geometry unavailable."
        )

    edges = np.unique(
        np.r_[
            0.0,
            geo["u_lo"],
            geo["interior"],
            geo["u_hi"],
            1.0,
        ]
    )

    edges = edges[
        (edges >= 0.0)
        & (edges <= 1.0)
    ]

    tested = 0

    for a, b in zip(
        edges[:-1],
        edges[1:],
    ):

        if b - a < 1e-6:
            continue

        pad = 0.02 * (
            b - a
        )

        u = np.linspace(
            a + pad,
            b - pad,
            9,
        )

        L = _logit(
            _curve_u(
                model,
                smooth_rows,
                u,
            )
        )

        assert np.all(
            np.isfinite(L)
        )

        d4 = np.abs(
            np.diff(
                L,
                n=4,
                axis=1,
            )
        )

        scale = np.maximum(
            np.abs(L).max(
                axis=1,
                keepdims=True,
            ),
            1.0,
        )

        assert np.all(
            d4 <= 1e-8 * scale
        ), (
            f"logit is not cubic on "
            f"[{a:.4f}, {b:.4f}]"
        )

        tested += 1

    assert tested >= 2, (
        "Expected at least two distinct knot intervals to test."
    )


def test_heterogeneous_response_shapes(model):
    df = _synthetic_frame(
        n=400,
        seed=11,
    )[REQUIRED_INFERENCE_COLS]

    grid = np.array([
        0.0,
        10.0,
        25.0,
        50.0,
        100.0,
    ])

    curves = model.predict_response_curve(
        df,
        grid,
    )

    uplift = (
        curves[:, -1]
        - curves[:, 0]
    )

    assert uplift.std() > 1e-3, (
        "All customers share one response curve."
    )

    assert np.ptp(uplift) > 1e-2, (
        "Response heterogeneity is implausibly narrow."
    )


def test_diminishing_returns_on_average(model):
    df = _synthetic_frame(
        n=800,
        seed=13,
    )[REQUIRED_INFERENCE_COLS]

    grid = np.array([
        0.0,
        20.0,
        70.0,
        100.0,
    ])

    c = model.predict_response_curve(
        df,
        grid,
    ).mean(axis=0)

    assert (
        c[1] - c[0]
    ) > (
        c[3] - c[2]
    ), (
        "Expected larger gains from 0->20% "
        "than from 70->100%."
    )


def test_response_curve_helper_shapes(
    model,
    rows,
):
    grid = [
        0,
        7.38,
        12.47,
        100,
    ]

    c = model.predict_response_curve(
        rows,
        grid,
    )

    assert c.shape == (
        len(rows),
        len(grid),
    )

    m = model.predict_response_curve(
        rows,
        grid,
        enforce_monotone=True,
    )

    assert np.all(
        np.diff(
            m,
            axis=1,
        ) >= -1e-12
    )


# =============================================================================
# INPUT HANDLING
# =============================================================================

def test_accepts_dict_and_broadcasts(
    model,
    rows,
):
    one = rows.iloc[
        0
    ].to_dict()

    p = model.predict_conversion_probability(
        one,
        18.25,
    )

    assert p.shape == (
        1,
    )

    per_row = model.predict_conversion_probability(
        rows,
        np.linspace(
            0,
            100,
            len(rows),
        ),
    )

    assert per_row.shape == (
        len(rows),
    )


@pytest.mark.parametrize(
    "bad",
    [
        -0.01,
        100.01,
        150.0,
        np.nan,
    ],
)
def test_out_of_range_discount_raises(
    model,
    rows,
    bad,
):
    with pytest.raises(ValueError):
        model.predict_conversion_probability(
            rows,
            bad,
        )


def test_missing_feature_column_raises(
    model,
    rows,
):
    broken = rows.drop(
        columns=[
            "cart_value"
        ]
    )

    with pytest.raises(KeyError):
        model.predict_conversion_probability(
            broken,
            10.0,
        )


def test_mismatched_discount_length_raises(
    model,
    rows,
):
    with pytest.raises(ValueError):
        model.predict_conversion_probability(
            rows,
            [
                0.0,
                5.0,
                10.0,
            ],
        )


# =============================================================================
# ECONOMICS
# =============================================================================

def test_expected_profit_matches_project_formula(
    model,
    rows,
):
    d = 12.47

    profit = model.expected_profit(
        rows,
        d,
    )

    p = model.predict_conversion_probability(
        rows,
        d,
    )

    V = rows[
        "cart_value"
    ].to_numpy(float)

    m = (
        rows[
            "margin_percentage"
        ].to_numpy(float)
        / 100.0
    )

    expected = (
        p
        * (
            V * (1 - d / 100.0)
            - V * (1 - m)
        )
    )

    np.testing.assert_allclose(
        profit,
        expected,
        rtol=1e-10,
    )


def test_full_discount_profit_is_negative_not_clamped(
    model,
    rows,
):
    """
    Per the brief: at 100% revenue is zero but cost remains,
    so expected profit must be negative.
    It must NOT be forced to zero.
    """
    profit = model.expected_profit(
        rows,
        100.0,
    )

    assert np.all(
        profit < 0
    )


# =============================================================================
# STAGE-3 BRIDGE
# =============================================================================

def test_legacy_arm_predictions_shape(
    model,
    rows,
):
    arms = model.legacy_arm_predictions(
        rows
    )

    assert set(arms) == {
        "pred_p0",
        "pred_p5",
        "pred_p10",
        "pred_p15",
    }

    for v in arms.values():

        assert v.shape == (
            len(rows),
        )

        assert np.all(
            (v > 0)
            & (v < 1)
        )