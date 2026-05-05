import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.project_paths import MODEL_RESULTS_DIR, PROCESSED_DATA_DIR


SPINE_PATH = PROCESSED_DATA_DIR / "weekly_macro_spine.csv"
DECOMPOSED_PATH = MODEL_RESULTS_DIR / "combined_mortgage_forecast_backtest.csv"
OUTPUT_DIR = MODEL_RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "direct_vs_decomposed_backtest.csv"
SUMMARY_PATH = OUTPUT_DIR / "direct_vs_decomposed_summary.csv"


DIRECT_ALPHA = 1000

DIRECT_MORTGAGE_FEATURES = [
    "MORTGAGE30US",
    "DGS10",
    "DGS2",
    "DGS3MO",
    "DGS5",
    "DGS30",
    "DFII10",
    "T10YIE",
    "FEDFUNDS",
    "BAA",
    "AAA",
    "NFCI",
    "VIXCLS",
    "UNRATE",
    "mortgage_spread",
    "yield_curve_10y_2y",
    "yield_curve_10y_3mo",
    "yield_curve_30y_10y",
    "credit_spread_baa_aaa",
    "credit_spread_baa_10y",
    "credit_spread_aaa_10y",
    "mortgage30us_chg_1w",
    "mortgage30us_chg_4w",
    "mortgage_spread_chg_1w",
    "mortgage_spread_chg_4w",
    "dgs10_chg_1w",
    "dgs10_chg_4w",
    "dgs10_roll_std_4w",
    "dgs10_roll_std_12w",
    "mortgage_spread_roll_std_4w",
    "mortgage_spread_roll_std_12w",
    "mortgage_spread_gap_vs_5y_avg",
    "vixcls_chg_1w",
    "vixcls_chg_4w",
    "vixcls_roll_std_4w",
    "vixcls_roll_std_12w",
    "nfci_chg_1w",
    "nfci_chg_4w",
    "nfci_roll_std_4w",
    "nfci_roll_std_12w",
]


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = "train"
    out.loc[out["week_date"] >= "2019-01-01", "split"] = "validation"
    out.loc[out["week_date"] >= "2023-01-01", "split"] = "test"
    return out


def metric_row(model_name, y_true, y_pred):
    return {
        "model_name": model_name,
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def main():
    spine = pd.read_csv(SPINE_PATH, parse_dates=["week_date"])
    decomposed = pd.read_csv(DECOMPOSED_PATH, parse_dates=["week_date"])

    # Direct target: next-week mortgage-rate change.
    spine = spine.sort_values("week_date").copy()
    spine["target_mortgage_rate_chg_1w"] = (
        spine["MORTGAGE30US"].shift(-1) - spine["MORTGAGE30US"]
    )

    base_cols = ["week_date", "target_mortgage_rate_chg_1w"]
    selected_cols = base_cols + DIRECT_MORTGAGE_FEATURES

    # remove duplicate column names while preserving order
    selected_cols = list(dict.fromkeys(selected_cols))

    model_df = spine[selected_cols].copy()

    model_df = model_df.dropna(subset=["target_mortgage_rate_chg_1w"] + DIRECT_MORTGAGE_FEATURES)
    model_df = add_split(model_df)

    X = model_df[DIRECT_MORTGAGE_FEATURES]
    y = model_df["target_mortgage_rate_chg_1w"]

    train_mask = model_df["split"] == "train"

    direct_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=DIRECT_ALPHA)),
        ]
    )

    direct_model.fit(X[train_mask], y[train_mask])

    model_df["pred_direct_mortgage_chg_1w"] = direct_model.predict(X)
    model_df["pred_direct_next_mortgage_rate"] = (
        model_df["MORTGAGE30US"] + model_df["pred_direct_mortgage_chg_1w"]
    )
    model_df["actual_next_mortgage_rate_direct_target"] = (
        model_df["MORTGAGE30US"] + model_df["target_mortgage_rate_chg_1w"]
    )

    compare = decomposed.merge(
        model_df[
            [
                "week_date",
                "pred_direct_mortgage_chg_1w",
                "pred_direct_next_mortgage_rate",
                "actual_next_mortgage_rate_direct_target",
            ]
        ],
        on="week_date",
        how="inner",
    )

    test = compare[compare["split"] == "test"].copy()

    summary = pd.DataFrame(
        [
            metric_row(
                "no_change_baseline",
                test["actual_next_mortgage_rate"],
                test["baseline_next_mortgage_rate"],
            ),
            metric_row(
                "direct_mortgage_model",
                test["actual_next_mortgage_rate"],
                test["pred_direct_next_mortgage_rate"],
            ),
            metric_row(
                "decomposed_10y_plus_spread_model",
                test["actual_next_mortgage_rate"],
                test["pred_next_mortgage_rate"],
            ),
        ]
    ).sort_values("rmse")

    compare.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nDirect vs decomposed mortgage forecast:")
    print(summary.to_string(index=False))

    ridge = direct_model.named_steps["ridge"]
    feature_importance = (
        pd.Series(ridge.coef_, index=DIRECT_MORTGAGE_FEATURES)
        .sort_values(key=abs, ascending=False)
    )

    print("\nTop 15 direct mortgage model features:")
    print(feature_importance.head(15).to_string())

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)

    print("\nRecent comparison rows:")
    print(
        test[
            [
                "week_date",
                "MORTGAGE30US",
                "actual_next_mortgage_rate",
                "baseline_next_mortgage_rate",
                "pred_direct_next_mortgage_rate",
                "pred_next_mortgage_rate",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
