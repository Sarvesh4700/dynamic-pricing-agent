"""
Stage 2: single CONTINUOUS discount-response model for the Dynamic Pricing Agent.

Estimates P(conversion | customer_features, discount_percentage) for any
discount_percentage in [0, 100].

ARCHITECTURE:
    logit P(convert | x, d) =
        h(x) + phi(u)'beta + (z(x) x [u, u^2])'gamma

where:

    u = log1p(d) / log1p(100)

The discount is treated as a real-valued continuous variable.

Run from project root:

    python src/continuous_response_model.py
    python src/continuous_response_model.py --fast
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG / SCHEMA
# =============================================================================

RANDOM_STATE = 42

MODEL_DIR = "models/uplift_continuous"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "continuous_response_model.joblib",
)
META_PATH = os.path.join(
    MODEL_DIR,
    "feature_metadata.json",
)
REPORT_PATH = os.path.join(
    MODEL_DIR,
    "training_report.json",
)

DATA_DIR_CANDIDATES = [
    "data/raw/continuous",
    os.path.join("..", "data", "raw", "continuous"),
    "data",
    ".",
]

CATEGORICAL_FEATURES = [
    "customer_type",
    "category",
    "device_type",
]

NUMERICAL_FEATURES = [
    "previous_purchases",
    "previous_abandons",
    "days_since_last_purchase",
    "customer_lifetime_value",
    "historical_discount_rate",
    "recent_discount_count",
    "days_since_last_discount",
    "cart_value",
    "items_count",
    "margin_percentage",
    "time_on_checkout_seconds",
    "pages_viewed",
    "payment_attempts",
    "hour",
    "day_of_week",
]

DISCOUNT_COL = "discount_percentage"
TARGET_COL = "converted"

FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
REQUIRED_INFERENCE_COLS = FEATURE_COLS

FORBIDDEN_COLUMNS = {
    "segment",
    "price_sensitivity",
    "base_logit_no_discount",
    "previous_abandons_at_eval",
    "true_probability_at_assigned_discount",
    "converted",
}

DISCOUNT_MIN = 0.0
DISCOUNT_MAX = 100.0

REFERENCE_DISCOUNT_GRID = [
    0,
    2.5,
    5,
    7.5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
]

ARBITRARY_DISCOUNTS = [
    7.38,
    12.47,
    18.25,
    33.33,
    57.91,
    91.2,
]

NEVER_SENTINEL = -1

SPLINE_NUMERIC = [
    "log_cart_value",
    "log_clv",
    "log_time_on_checkout",
    "log_days_since_last_purchase",
    "log_days_since_last_discount",
    "hist_discount_rate",
    "hour",
    "log_pages_viewed",
]

HETERO_NUMERIC = [
    "log_previous_purchases",
    "log_previous_abandons",
    "log_total_sessions",
    "abandon_share",
    "log_clv",
    "log_cart_value",
    "log_time_on_checkout",
    "hist_discount_rate",
    "log_recent_discount_count",
    "margin_frac",
]


# =============================================================================
# HELPERS
# =============================================================================

def _hr(title=""):
    print("\n" + "=" * 78)

    if title:
        print(title)
        print("=" * 78)


def saturating_coordinate(discount_pct):
    """
    u = log1p(d) / log1p(100)

    Maps discount d in [0,100] to u in [0,1].
    """
    d = np.clip(
        np.asarray(discount_pct, dtype=float),
        DISCOUNT_MIN,
        DISCOUNT_MAX,
    )

    return np.log1p(d) / np.log1p(100.0)


def _make_one_hot_encoder():
    """OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def _as_frame(X):
    """
    Accept:
        - pandas DataFrame
        - single dict
        - list of dicts
    """
    if isinstance(X, pd.DataFrame):
        return X

    if isinstance(X, dict):
        return pd.DataFrame([X])

    if isinstance(X, (list, tuple)):
        return pd.DataFrame(list(X))

    raise TypeError(
        f"Expected a pandas DataFrame, dict, or list of dicts; "
        f"got {type(X)!r}."
    )


# =============================================================================
# FEATURIZER
# =============================================================================

class DiscountResponseFeaturizer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Builds the full design matrix from raw observable columns + discount.

    Output blocks:

        [ one-hot cats
        | engineered numerics
        | numeric splines
        | u, u^2
        | discount spline basis
        | z(x) * u
        | z(x) * u^2
        ]
    """

    def __init__(
        self,
        n_discount_knots=6,
        n_numeric_knots=5,
        degree=3,
    ):
        self.n_discount_knots = n_discount_knots
        self.n_numeric_knots = n_numeric_knots
        self.degree = degree

    # -------------------------------------------------------------------------
    # Engineered numeric frame
    # -------------------------------------------------------------------------

    def _raw_numeric(self, X):

        def g(col):
            if col not in X.columns:
                raise KeyError(
                    f"Required feature column '{col}' is missing. "
                    f"Expected all of: {REQUIRED_INFERENCE_COLS}"
                )

            return pd.to_numeric(
                X[col],
                errors="coerce",
            ).to_numpy(dtype=float)

        prev_p = g("previous_purchases")
        prev_a = g("previous_abandons")
        dslp = g("days_since_last_purchase")
        dsld = g("days_since_last_discount")

        sessions = (
            np.clip(prev_p, 0, None)
            + np.clip(prev_a, 0, None)
        )

        attempts = np.clip(
            g("payment_attempts"),
            0,
            None,
        )

        hour = np.clip(
            g("hour"),
            0,
            23,
        )

        dow = np.clip(
            g("day_of_week"),
            0,
            6,
        )

        out = {
            "log_previous_purchases": np.log1p(
                np.clip(prev_p, 0, None)
            ),

            "log_previous_abandons": np.log1p(
                np.clip(prev_a, 0, None)
            ),

            "log_total_sessions": np.log1p(
                sessions
            ),

            "abandon_share": np.where(
                sessions > 0,
                np.clip(prev_a, 0, None)
                / np.maximum(sessions, 1.0),
                0.0,
            ),

            "no_prior_purchase": (
                dslp <= NEVER_SENTINEL
            ).astype(float),

            "log_days_since_last_purchase": np.log1p(
                np.clip(
                    np.where(
                        dslp <= NEVER_SENTINEL,
                        0.0,
                        dslp,
                    ),
                    0,
                    None,
                )
            ),

            "no_prior_discount": (
                dsld <= NEVER_SENTINEL
            ).astype(float),

            "log_days_since_last_discount": np.log1p(
                np.clip(
                    np.where(
                        dsld <= NEVER_SENTINEL,
                        0.0,
                        dsld,
                    ),
                    0,
                    None,
                )
            ),

            "log_clv": np.log1p(
                np.clip(
                    g("customer_lifetime_value"),
                    0,
                    None,
                )
            ),

            "log_cart_value": np.log1p(
                np.clip(
                    g("cart_value"),
                    0,
                    None,
                )
            ),

            "log_items_count": np.log1p(
                np.clip(
                    g("items_count"),
                    0,
                    None,
                )
            ),

            "margin_frac": (
                g("margin_percentage") / 100.0
            ),

            "log_time_on_checkout": np.log1p(
                np.clip(
                    g("time_on_checkout_seconds"),
                    0,
                    None,
                )
            ),

            "log_pages_viewed": np.log1p(
                np.clip(
                    g("pages_viewed"),
                    0,
                    None,
                )
            ),

            "payment_attempts": attempts,

            "extra_payment_attempts": np.clip(
                attempts - 1.0,
                0,
                None,
            ),

            "hist_discount_rate": (
                np.clip(
                    g("historical_discount_rate"),
                    0,
                    100,
                ) / 100.0
            ),

            "log_recent_discount_count": np.log1p(
                np.clip(
                    g("recent_discount_count"),
                    0,
                    None,
                )
            ),

            "hour": hour,

            "hour_sin": np.sin(
                2.0 * np.pi * hour / 24.0
            ),

            "hour_cos": np.cos(
                2.0 * np.pi * hour / 24.0
            ),

            "day_of_week": dow,

            "is_weekend": (
                dow >= 5
            ).astype(float),
        }

        return pd.DataFrame(
            out,
            index=X.index,
        )

    def _discount_vector(self, X):

        if DISCOUNT_COL not in X.columns:
            raise KeyError(
                f"'{DISCOUNT_COL}' must be present on the "
                f"frame passed to the featurizer "
                f"(the model class attaches it for you)."
            )

        return np.clip(
            pd.to_numeric(
                X[DISCOUNT_COL],
                errors="coerce",
            ).to_numpy(dtype=float),
            DISCOUNT_MIN,
            DISCOUNT_MAX,
        )

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    def fit(self, X, y=None):

        X = _as_frame(X)

        leaked = (
            set(X.columns) & FORBIDDEN_COLUMNS
        ) - {TARGET_COL}

        if leaked:
            raise ValueError(
                "Ground-truth / evaluation-only columns "
                f"reached the featurizer: {sorted(leaked)}. "
                "These must never be features."
            )

        self.ohe_ = _make_one_hot_encoder()

        self.ohe_.fit(
            X[CATEGORICAL_FEATURES].astype(str)
        )

        self.cat_names_ = list(
            self.ohe_.get_feature_names_out(
                CATEGORICAL_FEATURES
            )
        )

        num = self._raw_numeric(X)

        self.numeric_names_ = list(
            num.columns
        )

        self.numeric_medians_ = (
            num.median(
                numeric_only=True
            ).to_dict()
        )

        num = num.fillna(
            value=self.numeric_medians_
        )

        self.spline_cols_ = [
            c
            for c in SPLINE_NUMERIC
            if c in num.columns
        ]

        self.numeric_spline_ = SplineTransformer(
            n_knots=self.n_numeric_knots,
            degree=self.degree,
            knots="uniform",
            extrapolation="constant",
            include_bias=False,
        )

        self.numeric_spline_.fit(
            num[self.spline_cols_].to_numpy(
                dtype=float
            )
        )

        u = saturating_coordinate(
            self._discount_vector(X)
        ).reshape(-1, 1)

        self.discount_spline_ = SplineTransformer(
            n_knots=self.n_discount_knots,
            degree=self.degree,
            knots="uniform",
            extrapolation="constant",
            include_bias=False,
        )

        self.discount_spline_.fit(u)

        self.hetero_cols_ = [
            c
            for c in HETERO_NUMERIC
            if c in num.columns
        ]

        self.hetero_names_ = (
            self.cat_names_
            + list(self.hetero_cols_)
        )

        n_num_spline = (
            self.numeric_spline_
            .transform(
                num[
                    self.spline_cols_
                ].to_numpy(
                    dtype=float
                )[:1]
            )
            .shape[1]
        )

        n_d_spline = (
            self.discount_spline_
            .transform(u[:1])
            .shape[1]
        )

        self.feature_names_out_ = (
            [
                f"cat__{c}"
                for c in self.cat_names_
            ]
            + [
                f"num__{c}"
                for c in self.numeric_names_
            ]
            + [
                f"numspline__{i}"
                for i in range(n_num_spline)
            ]
            + [
                "disc__u",
                "disc__u2",
            ]
            + [
                f"disc__spline_{i}"
                for i in range(n_d_spline)
            ]
            + [
                f"het__{n}__x_u"
                for n in self.hetero_names_
            ]
            + [
                f"het__{n}__x_u2"
                for n in self.hetero_names_
            ]
        )

        self.n_features_out_ = len(
            self.feature_names_out_
        )

        return self

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------

    def transform(self, X):

        X = _as_frame(X)

        cat = self.ohe_.transform(
            X[CATEGORICAL_FEATURES].astype(str)
        )

        num = self._raw_numeric(X).fillna(
            value=self.numeric_medians_
        )

        num = num[
            self.numeric_names_
        ]

        num_lin = num.to_numpy(
            dtype=float
        )

        num_spl = self.numeric_spline_.transform(
            num[
                self.spline_cols_
            ].to_numpy(dtype=float)
        )

        u = saturating_coordinate(
            self._discount_vector(X)
        ).reshape(-1, 1)

        u2 = u ** 2

        d_spl = self.discount_spline_.transform(u)

        z = np.hstack(
            [
                cat,
                num[
                    self.hetero_cols_
                ].to_numpy(dtype=float),
            ]
        )

        design = np.hstack(
            [
                cat,
                num_lin,
                num_spl,
                u,
                u2,
                d_spl,
                z * u,
                z * u2,
            ]
        )

        if design.shape[1] != self.n_features_out_:
            raise RuntimeError(
                f"Design matrix width {design.shape[1]} "
                f"!= fitted {self.n_features_out_}. "
                "Feature schema drift."
            )

        return np.nan_to_num(
            design,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return np.asarray(
            self.feature_names_out_,
            dtype=object,
        )


# =============================================================================
# MODEL
# =============================================================================

class ContinuousResponseModel:
    """
    One model with continuous discount input.

    Primary interface:

        predict_conversion_probability(...)
        predict_response_curve(...)
        expected_profit(...)
        legacy_arm_predictions(...)
    """

    model_version = "continuous-response-1.0.0"

    def __init__(
        self,
        C=1.0,
        n_discount_knots=6,
        n_numeric_knots=5,
        degree=3,
        max_iter=3000,
        random_state=RANDOM_STATE,
    ):
        self.C = C
        self.n_discount_knots = n_discount_knots
        self.n_numeric_knots = n_numeric_knots
        self.degree = degree
        self.max_iter = max_iter
        self.random_state = random_state

        self.pipeline_ = None
        self.fitted_ = False
        self.training_info_ = {}

    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------

    def _build_pipeline(self):

        return Pipeline(
            [
                (
                    "features",
                    DiscountResponseFeaturizer(
                        n_discount_knots=(
                            self.n_discount_knots
                        ),
                        n_numeric_knots=(
                            self.n_numeric_knots
                        ),
                        degree=self.degree,
                    ),
                ),
                (
                    "scale",
                    StandardScaler(
                        with_mean=True,
                        with_std=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=self.C,
                        solver="lbfgs",
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_discount(d):

        arr = np.asarray(
            d,
            dtype=float,
        ).ravel()

        if arr.size == 0:
            raise ValueError(
                "discount_percentage is empty."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "discount_percentage contains NaN/inf."
            )

        if (
            np.any(arr < DISCOUNT_MIN)
            or np.any(arr > DISCOUNT_MAX)
        ):
            raise ValueError(
                "discount_percentage must lie in "
                f"[{DISCOUNT_MIN}, {DISCOUNT_MAX}]; "
                f"got min={arr.min()}, max={arr.max()}."
            )

        return arr

    def _prepare(
        self,
        features,
        discount_percentage,
    ):

        if not self.fitted_:
            raise RuntimeError(
                "Model is not fitted. "
                "Call fit() or load()."
            )

        X = _as_frame(
            features
        ).copy()

        missing = [
            c
            for c in REQUIRED_INFERENCE_COLS
            if c not in X.columns
        ]

        if missing:
            raise KeyError(
                "Missing required feature columns "
                f"at inference: {missing}"
            )

        d = self._validate_discount(
            discount_percentage
        )

        n = len(X)

        if d.size == 1:
            d = np.full(
                n,
                d[0],
                dtype=float,
            )

        elif d.size != n:
            raise ValueError(
                f"discount_percentage has length "
                f"{d.size} but {n} feature rows "
                "were supplied. Pass one scalar, "
                "or one value per row, or use "
                "predict_response_curve() for a "
                "full grid per row."
            )

        X[DISCOUNT_COL] = d

        return X

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    def fit(
        self,
        tx_df,
        y=None,
    ):

        df = _as_frame(tx_df)

        if y is None:

            if TARGET_COL not in df.columns:
                raise KeyError(
                    f"Target '{TARGET_COL}' "
                    "not found and y not given."
                )

            y = df[TARGET_COL].to_numpy()

        y = np.asarray(y).astype(int).ravel()

        if not set(
            np.unique(y)
        ).issubset({0, 1}):

            raise ValueError(
                "Target must be binary 0/1."
            )

        cols = (
            REQUIRED_INFERENCE_COLS
            + [DISCOUNT_COL]
        )

        missing = [
            c
            for c in cols
            if c not in df.columns
        ]

        if missing:
            raise KeyError(
                "Training frame is missing "
                f"columns: {missing}"
            )

        X = df[cols].copy()

        leaked = (
            set(X.columns)
            & FORBIDDEN_COLUMNS
        )

        assert not leaked, (
            "Leakage guard tripped: "
            f"{sorted(leaked)}"
        )

        self.pipeline_ = (
            self._build_pipeline()
        )

        self.pipeline_.fit(
            X,
            y,
        )

        self.fitted_ = True

        self.training_info_.update(
            {
                "n_train_rows": int(
                    len(X)
                ),

                "n_design_columns": int(
                    self.pipeline_
                    .named_steps[
                        "features"
                    ]
                    .n_features_out_
                ),

                "train_conversion_rate": float(
                    y.mean()
                ),

                "C": float(self.C),

                "n_discount_knots": int(
                    self.n_discount_knots
                ),

                "n_numeric_knots": int(
                    self.n_numeric_knots
                ),

                "fitted_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }
        )

        return self

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict_conversion_probability(
        self,
        features,
        discount_percentage,
    ):
        """
        P(conversion | features, discount_percentage)
        for any d in [0,100].
        """

        X = self._prepare(
            features,
            discount_percentage,
        )

        p = (
            self.pipeline_
            .predict_proba(X)[:, 1]
        )

        return np.clip(
            p,
            1e-6,
            1 - 1e-6,
        )

    def predict_response_curve(
        self,
        features,
        discount_grid=None,
        enforce_monotone=False,
    ):
        """
        Returns a response curve for every row.

        Shape:
            (n_rows, n_grid)

        enforce_monotone=False by default.
        """

        grid = np.asarray(
            (
                REFERENCE_DISCOUNT_GRID
                if discount_grid is None
                else discount_grid
            ),
            dtype=float,
        ).ravel()

        self._validate_discount(
            grid
        )

        X = _as_frame(
            features
        )

        out = np.empty(
            (
                len(X),
                grid.size,
            ),
            dtype=float,
        )

        for j, d in enumerate(grid):

            out[:, j] = (
                self.predict_conversion_probability(
                    X,
                    d,
                )
            )

        if enforce_monotone:

            order = np.argsort(
                grid
            )

            tmp = out[
                :,
                order,
            ]

            np.maximum.accumulate(
                tmp,
                axis=1,
                out=tmp,
            )

            out[
                :,
                order,
            ] = tmp

        return out

    # -------------------------------------------------------------------------
    # Expected profit
    # -------------------------------------------------------------------------

    def expected_profit(
        self,
        features,
        discount_percentage,
    ):
        """
        Existing project convention:

            cost = V * (1-m)
            price = V * (1-d/100)
            profit_if_converted = price - cost
            expected_profit = P(convert) * profit_if_converted
        """

        X = _as_frame(
            features
        )

        p = (
            self.predict_conversion_probability(
                X,
                discount_percentage,
            )
        )

        V = pd.to_numeric(
            X["cart_value"],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        m = (
            pd.to_numeric(
                X["margin_percentage"],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
            / 100.0
        )

        d = np.asarray(
            discount_percentage,
            dtype=float,
        ).ravel()

        if d.size == 1:
            d = np.full(
                len(X),
                d[0],
                dtype=float,
            )

        cost = V * (
            1.0 - m
        )

        price_after_discount = (
            V * (
                1.0 - d / 100.0
            )
        )

        return (
            p
            * (
                price_after_discount
                - cost
            )
        )

    # -------------------------------------------------------------------------
    # Legacy bridge
    # -------------------------------------------------------------------------

    def legacy_arm_predictions(
        self,
        features,
        arms=(0, 5, 10, 15),
    ):
        """
        Stage-3 bridge.

        Emits:
            pred_p0
            pred_p5
            pred_p10
            pred_p15
        """

        X = _as_frame(
            features
        )

        return {
            f"pred_p{int(a) if float(a).is_integer() else a}":
                self.predict_conversion_probability(
                    X,
                    float(a),
                )
            for a in arms
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(
        self,
        model_path=MODEL_PATH,
        meta_path=META_PATH,
    ):
        """
        Save the fitted model and metadata.

        Includes a safety check to prevent accidentally pickling
        custom classes under the '__main__' module.
        """

        if not self.fitted_:
            raise RuntimeError(
                "Refusing to save an unfitted model."
            )

        # ---------------------------------------------------------------------
        # Pickle safety guard
        # ---------------------------------------------------------------------
        #
        # If this file is executed directly, Python can register classes
        # under "__main__". joblib would then save them as:
        #
        #     __main__.ContinuousResponseModel
        #
        # Another Python process may then fail to load the artifact.
        #
        # The bootstrap at the bottom of this file imports the module using
        # its real name:
        #
        #     src.continuous_response_model
        #
        # so normally these checks should pass.
        # ---------------------------------------------------------------------

        objects_to_check = [
            self,
            self.pipeline_.named_steps["features"],
        ]

        for obj in objects_to_check:

            module_name = (
                type(obj).__module__
            )

            if module_name == "__main__":
                raise RuntimeError(
                    f"{type(obj).__name__} is defined in "
                    "'__main__', so joblib would pickle it "
                    "under that path. Another process may "
                    "fail to load this artifact.\n\n"
                    "Launch training through the bootstrap "
                    "at the bottom of "
                    "src/continuous_response_model.py, "
                    "which imports this module as "
                    "'src.continuous_response_model'."
                )

            if module_name != (
                "src.continuous_response_model"
            ):
                print(
                    "  [warning] "
                    f"{type(obj).__name__} is pickling "
                    f"under module '{module_name}'. "
                    "Loaders must be able to import "
                    "that exact module path."
                )

        # ---------------------------------------------------------------------
        # Save model
        # ---------------------------------------------------------------------

        os.makedirs(
            os.path.dirname(
                model_path
            ) or ".",
            exist_ok=True,
        )

        joblib.dump(
            self,
            model_path,
        )

        # ---------------------------------------------------------------------
        # Leakage guard
        # ---------------------------------------------------------------------

        assert not (
            set(FEATURE_COLS)
            & FORBIDDEN_COLUMNS
        ), (
            "Leakage guard: forbidden column "
            "present in saved feature list."
        )

        # ---------------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------------

        meta = {
            "model_version": self.model_version,

            "pickled_class_module": (
                type(self).__module__
            ),

            "architecture": (
                "structured logistic response model: "
                "additive spline baseline "
                "h(x) + cubic B-spline basis in "
                "u=log1p(d)/log1p(100) + "
                "z(x) x [u,u^2] heterogeneity interactions"
            ),

            "treatment_col": DISCOUNT_COL,

            "treatment_range": [
                DISCOUNT_MIN,
                DISCOUNT_MAX,
            ],

            "treatment_is_continuous": True,

            "target_col": TARGET_COL,

            "feature_cols": FEATURE_COLS,

            "categorical_features": (
                CATEGORICAL_FEATURES
            ),

            "numerical_features": (
                NUMERICAL_FEATURES
            ),

            "required_inference_cols": (
                REQUIRED_INFERENCE_COLS
            ),

            "forbidden_columns": sorted(
                FORBIDDEN_COLUMNS
            ),

            "reference_discount_grid": (
                REFERENCE_DISCOUNT_GRID
            ),

            "random_state": self.random_state,

            "training_info": (
                self.training_info_
            ),

            "notes": (
                "Loading requires "
                "src.continuous_response_model "
                "to be importable because the "
                "featurizer is a custom transformer. "
                "Old discrete artifacts under "
                "models/uplift/ are untouched."
            ),
        }

        with open(
            meta_path,
            "w",
        ) as f:

            json.dump(
                meta,
                f,
                indent=2,
                default=str,
            )

        print(
            f"  [saved model]    {model_path}"
        )

        print(
            f"  [saved metadata] {meta_path}"
        )

        return model_path

    @staticmethod
    def load(
        model_path=MODEL_PATH,
    ):
        if not os.path.exists(
            model_path
        ):
            raise FileNotFoundError(
                f"No model at {model_path}. "
                "Train it first:\n"
                "    python src/continuous_response_model.py"
            )

        obj = joblib.load(
            model_path
        )

        if not isinstance(
            obj,
            ContinuousResponseModel,
        ):
            raise TypeError(
                f"{model_path} did not contain "
                "a ContinuousResponseModel."
            )

        return obj


# =============================================================================
# DATA LOADING
# =============================================================================

def resolve_data_dir(
    explicit=None,
):

    candidates = (
        [explicit]
        if explicit
        else list(DATA_DIR_CANDIDATES)
    )

    for d in candidates:

        if d and os.path.exists(
            os.path.join(
                d,
                "transactions_train.csv",
            )
        ):
            return d

    raise FileNotFoundError(
        "Could not find transactions_train.csv. "
        f"Tried: {[c for c in candidates if c]}. "
        "Pass --data-dir explicitly."
    )


def load_continuous_data(
    data_dir=None,
):

    data_dir = resolve_data_dir(
        data_dir
    )

    tx_train = pd.read_csv(
        os.path.join(
            data_dir,
            "transactions_train.csv",
        ),
        parse_dates=["timestamp"],
    )

    tx_test = pd.read_csv(
        os.path.join(
            data_dir,
            "transactions_test.csv",
        ),
        parse_dates=["timestamp"],
    )

    print(
        f"Data dir: {data_dir}"
    )

    print(
        f"  train: {len(tx_train):,} rows, "
        f"{tx_train['customer_id'].nunique():,} customers"
    )

    print(
        f"  test:  {len(tx_test):,} rows, "
        f"{tx_test['customer_id'].nunique():,} customers"
    )

    overlap = (
        set(tx_train["customer_id"])
        & set(tx_test["customer_id"])
    )

    assert not overlap, (
        "Customer overlap between train/test: "
        f"{len(overlap)}"
    )

    for name, df in (
        ("train", tx_train),
        ("test", tx_test),
    ):

        d = df[
            DISCOUNT_COL
        ]

        assert (
            d.min() >= DISCOUNT_MIN
            and d.max() <= DISCOUNT_MAX
        ), (
            f"{name}: discount outside [0,100]"
        )

        assert set(
            df[TARGET_COL].unique()
        ) <= {0, 1}, (
            f"{name}: target not binary"
        )

        leaked = (
            set(df.columns)
            & (
                FORBIDDEN_COLUMNS
                - {TARGET_COL}
            )
        )

        assert not leaked, (
            f"{name}: unexpected "
            f"ground-truth columns present: "
            f"{leaked}"
        )

        print(
            f"  {name}: "
            f"unique discounts={d.nunique():,}, "
            f"min={d.min():.4f}, "
            f"max={d.max():.4f}, "
            f"conv={df[TARGET_COL].mean():.4f}"
        )

    return (
        tx_train,
        tx_test,
        data_dir,
    )


# =============================================================================
# CV SWEEP
# =============================================================================

def select_regularization(
    tx_train,
    c_grid=(
        0.05,
        0.2,
        1.0,
        5.0,
    ),
    n_splits=4,
):

    X = tx_train[
        REQUIRED_INFERENCE_COLS
        + [DISCOUNT_COL]
    ]

    y = (
        tx_train[TARGET_COL]
        .to_numpy()
        .astype(int)
    )

    groups = (
        tx_train["customer_id"]
        .to_numpy()
    )

    gkf = GroupKFold(
        n_splits=n_splits
    )

    rows = []

    for C in c_grid:

        losses = []
        aucs = []

        for tr, va in gkf.split(
            X,
            y,
            groups,
        ):

            m = ContinuousResponseModel(
                C=C
            )

            m.pipeline_ = (
                m._build_pipeline()
            )

            m.pipeline_.fit(
                X.iloc[tr],
                y[tr],
            )

            m.fitted_ = True

            p = np.clip(
                m.pipeline_
                .predict_proba(
                    X.iloc[va]
                )[:, 1],
                1e-6,
                1 - 1e-6,
            )

            losses.append(
                log_loss(
                    y[va],
                    p,
                    labels=[0, 1],
                )
            )

            aucs.append(
                roc_auc_score(
                    y[va],
                    p,
                )
            )

        rows.append(
            {
                "C": C,
                "cv_log_loss": float(
                    np.mean(losses)
                ),
                "cv_roc_auc": float(
                    np.mean(aucs)
                ),
            }
        )

        print(
            f"  C={C:<6} "
            f"cv_log_loss="
            f"{rows[-1]['cv_log_loss']:.5f}  "
            f"cv_roc_auc="
            f"{rows[-1]['cv_roc_auc']:.4f}"
        )

    table = (
        pd.DataFrame(rows)
        .sort_values("cv_log_loss")
        .reset_index(drop=True)
    )

    best_C = float(
        table.loc[0, "C"]
    )

    print(
        f"  -> selected C={best_C}"
    )

    return (
        best_C,
        table,
    )


# =============================================================================
# FACTUAL METRICS
# =============================================================================

def factual_metrics(
    model,
    tx_df,
    label="test",
):

    y = (
        tx_df[TARGET_COL]
        .to_numpy()
        .astype(int)
    )

    p = (
        model.predict_conversion_probability(
            tx_df,
            tx_df[DISCOUNT_COL].to_numpy(),
        )
    )

    out = {
        "split": label,
        "n_rows": int(
            len(tx_df)
        ),
        "conversion_rate": float(
            y.mean()
        ),
        "mean_predicted": float(
            p.mean()
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                p,
            )
        ),
        "log_loss": float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y,
                p,
            )
        ),
    }

    for k, v in out.items():
        print(
            f"  {k}: {v}"
        )

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Train the continuous "
            "discount-response model "
            "(Stage 2)."
        )
    )

    ap.add_argument(
        "--data-dir",
        default=None,
    )

    ap.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip the grouped-CV sweep; "
            "use C=1.0."
        ),
    )

    ap.add_argument(
        "--C",
        type=float,
        default=None,
        help=(
            "Force a specific C and "
            "skip the sweep."
        ),
    )

    args = ap.parse_args()

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    _hr(
        "LOAD CONTINUOUS DATASET"
    )

    (
        tx_train,
        tx_test,
        data_dir,
    ) = load_continuous_data(
        args.data_dir
    )

    _hr(
        "LEAKAGE GUARD"
    )

    print(
        f"Feature columns used "
        f"({len(FEATURE_COLS)}): "
        f"{FEATURE_COLS}"
    )

    print(
        f"Treatment passed separately: "
        f"{DISCOUNT_COL}"
    )

    print(
        f"Forbidden (evaluation-only): "
        f"{sorted(FORBIDDEN_COLUMNS)}"
    )

    assert not (
        set(FEATURE_COLS)
        & FORBIDDEN_COLUMNS
    )

    print(
        "OK: no ground-truth column is "
        "reachable from the feature set."
    )

    cv_table = None

    if args.C is not None:

        best_C = args.C

        print(
            f"\nUsing forced C={best_C}"
        )

    elif args.fast:

        best_C = 1.0

        print(
            "\n--fast: using C=1.0 "
            "without a sweep."
        )

    else:

        _hr(
            "REGULARIZATION SWEEP "
            "(GroupKFold on customer_id)"
        )

        (
            best_C,
            cv_table,
        ) = select_regularization(
            tx_train
        )

    _hr(
        "FIT ON FULL TRAINING SET"
    )

    model = ContinuousResponseModel(
        C=best_C
    )

    model.fit(
        tx_train
    )

    print(
        "  design columns: "
        f"{model.training_info_['n_design_columns']}"
    )

    print(
        "  train rows:     "
        f"{model.training_info_['n_train_rows']:,}"
    )

    _hr(
        "FACTUAL METRICS (train)"
    )

    train_m = factual_metrics(
        model,
        tx_train,
        "train",
    )

    _hr(
        "FACTUAL METRICS "
        "(held-out test)"
    )

    test_m = factual_metrics(
        model,
        tx_test,
        "test",
    )

    _hr(
        "SANITY: MEAN PREDICTED "
        "RESPONSE CURVE ON TEST CUSTOMERS"
    )

    curves = (
        model.predict_response_curve(
            tx_test,
            REFERENCE_DISCOUNT_GRID,
        )
    )

    mean_curve = (
        curves.mean(axis=0)
    )

    for d, v in zip(
        REFERENCE_DISCOUNT_GRID,
        mean_curve,
    ):

        print(
            f"  {d:>6}% -> {v:.4f}"
        )

    deltas = np.diff(
        mean_curve
    )

    print(
        "  non-decreasing on the grid: "
        f"{bool(np.all(deltas >= -1e-9))}"
    )

    gain_low = (
        mean_curve[
            REFERENCE_DISCOUNT_GRID.index(20)
        ]
        - mean_curve[0]
    )

    gain_high = (
        mean_curve[-1]
        - mean_curve[
            REFERENCE_DISCOUNT_GRID.index(70)
        ]
    )

    print(
        f"  gain 0->20%: {gain_low:.4f} | "
        f"gain 70->100%: {gain_high:.4f} | "
        f"diminishing returns: "
        f"{gain_low > gain_high}"
    )

    _hr(
        "ARBITRARY-DISCOUNT SMOKE TEST"
    )

    sample = tx_test.head(3)

    for d in ARBITRARY_DISCOUNTS:

        p = (
            model.predict_conversion_probability(
                sample,
                d,
            )
        )

        print(
            f"  d={d:>6}% -> "
            + ", ".join(
                f"{v:.4f}"
                for v in p
            )
        )

    _hr(
        "SAVE ARTIFACTS"
    )

    model.save()

    report = {
        "data_dir": data_dir,

        "selected_C": best_C,

        "cv_table": (
            cv_table.to_dict(
                orient="records"
            )
            if cv_table is not None
            else None
        ),

        "train_metrics": train_m,

        "test_metrics": test_m,

        "mean_predicted_response_curve": dict(
            zip(
                map(
                    str,
                    REFERENCE_DISCOUNT_GRID,
                ),
                map(
                    float,
                    mean_curve,
                ),
            )
        ),

        "mean_curve_non_decreasing": bool(
            np.all(
                deltas >= -1e-9
            )
        ),

        "diminishing_returns": bool(
            gain_low > gain_high
        ),

        "training_info": (
            model.training_info_
        ),
    }

    with open(
        REPORT_PATH,
        "w",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    print(
        f"  [saved report]   "
        f"{REPORT_PATH}"
    )

    _hr(
        "DONE"
    )

    print(
        "Next: python src/evaluate_continuous_model.py"
    )


# =============================================================================
# DIRECT EXECUTION / PICKLE SAFETY BOOTSTRAP
# =============================================================================

if __name__ == "__main__":

    # Pickle records a class by its module path.
    #
    # Executing this file directly would normally make classes appear to
    # belong to "__main__". Instead, import the module using its real
    # package path and delegate to its main().
    #
    # This ensures the saved custom classes are recorded as:
    #
    #     src.continuous_response_model
    #
    # rather than:
    #
    #     __main__
    #
    # This block does not recurse because the imported module has a different
    # __name__.

    import os as _os
    import sys as _sys

    _PROJECT_ROOT = _os.path.dirname(
        _os.path.dirname(
            _os.path.abspath(__file__)
        )
    )

    if _PROJECT_ROOT not in _sys.path:
        _sys.path.insert(
            0,
            _PROJECT_ROOT,
        )

    from src.continuous_response_model import (
        main as _main,
    )

    _main()