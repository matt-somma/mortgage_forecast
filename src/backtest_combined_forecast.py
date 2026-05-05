import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from project_paths import MODEL_RESULTS_DIR, MODELING_DATA_DIR


DGS10_PATH = MODELING_DATA_DIR / "dgs10_long_1w.csv"
SPREAD_PATH = MODELING_DATA_DIR / "spread_long_1w.csv"
OUTPUT_DIR = MODEL_RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "combined_mortgage_forecast_backtest.csv"
SUMMARY_PATH = OUTPUT_DIR / "combined_mortgage_forecast_summary.csv"


DGS10_ALPHA = 1000
SPREAD_ALPHA = 500

DGS10_FEATURES = [
    "DGS10",
    "DGS2",
    "DGS3MO",
    "DGS5",
    "DGS30",
    "DFII10",
    "T10YIE",
    "yield_curve_10y_2y",
    "yield_curve_10y_3mo",
    "yield_curve_30y_10y",
    "dgs10_chg_1w",
    "dgs10_chg_4w",
    "dgs2_chg_1w",
    "dgs5_chg_1w",
    "dgs30_chg_1w",
    "dgs10_roll_std_4w",
    "dgs10_roll_std_12w",
    "VIXCLS",
    "vixcls_chg_1w",
    "vixcls_roll_std_4w",
]

SPREAD_FEATURES = [
    "mortgage_spread",
    "MORTGAGE30US",
    "DGS10",
    "DGS2",
    "DGS3MO",
    "DGS5",
    "DGS30",
    "FEDFUNDS",
    "BAA",
    "AAA",
    "NFCI",
    "VIXCLS",
    "UNRATE",
    "CPIAUCSL",
    "yield_curve_10y_2y",
    "yield_curve_10y_3mo",
    "yield_curve_30y_10y",
    "credit_spread_baa_aaa",
    "credit_spread_baa_10y",
    "credit_spread_aaa_10y",
    "mortgage_spread_chg_1w",
    "mortgage_spread_chg_4w",
    "mortgage_spread_roll_mean_4w",
    "mortgage_spread_roll_mean_12w",
    "mortgage_spread_roll_std_4w",
    "mortgage_spread_roll_std_12w",
    "mortgage_spread_roll_mean_5y",
    "mortgage_spread_gap_vs_5y_avg",
    "dgs10_chg_1w",
    "dgs10_chg_4w",
    "dgs10_roll_std_4w",
    "dgs10_roll_std_12w",
    "vixcls_chg_1w",
    "vixcls_chg_4w",
    "vixcls_roll_std_4w",
    "vixcls_roll_std_12w",
    "nfci_chg_1w",
    "nfci_chg_4w",
    "nfci_roll_mean_4w",
    "nfci_roll_mean_12w",
    "nfci_roll_std_4w",
    "nfci_roll_std_12w",
]


def fit_ridge(df: pd.DataFrame, features: list[str], alpha: float):
    X = df[features]
    y = df["target"]

    train_mask = df["split"] == "train"

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )

    model.fit(X[train_mask], y[train_mask])

    return model


def metric_row(model_name: str, y_true, y_pred):
    return {
        "model_name": model_name,
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def main():
    dgs10_df = pd.read_csv(DGS10_PATH, parse_dates=["week_date"])
    spread_df = pd.read_csv(SPREAD_PATH, parse_dates=["week_date"])

    dgs10_model = fit_ridge(dgs10_df, DGS10_FEATURES, DGS10_ALPHA)
    spread_model = fit_ridge(spread_df, SPREAD_FEATURES, SPREAD_ALPHA)

    dgs10_pred_df = dgs10_df[["week_date", "split", "DGS10", "target"]].copy()
    dgs10_pred_df = dgs10_pred_df.rename(columns={"target": "actual_dgs10_chg_1w"})
    dgs10_pred_df["pred_dgs10_chg_1w"] = dgs10_model.predict(dgs10_df[DGS10_FEATURES])

    spread_pred_df = spread_df[
        ["week_date", "split", "MORTGAGE30US", "DGS10", "mortgage_spread", "target"]
    ].copy()
    spread_pred_df = spread_pred_df.rename(columns={"target": "actual_spread_chg_1w"})
    spread_pred_df["pred_spread_chg_1w"] = spread_model.predict(spread_df[SPREAD_FEATURES])

    backtest = spread_pred_df.merge(
        dgs10_pred_df[["week_date", "actual_dgs10_chg_1w", "pred_dgs10_chg_1w"]],
        on="week_date",
        how="inner",
    )

    backtest["actual_next_dgs10"] = backtest["DGS10"] + backtest["actual_dgs10_chg_1w"]
    backtest["pred_next_dgs10"] = backtest["DGS10"] + backtest["pred_dgs10_chg_1w"]

    backtest["actual_next_spread"] = (
        backtest["mortgage_spread"] + backtest["actual_spread_chg_1w"]
    )
    backtest["pred_next_spread"] = (
        backtest["mortgage_spread"] + backtest["pred_spread_chg_1w"]
    )

    backtest["actual_next_mortgage_rate"] = (
        backtest["actual_next_dgs10"] + backtest["actual_next_spread"]
    )

    backtest["pred_next_mortgage_rate"] = (
        backtest["pred_next_dgs10"] + backtest["pred_next_spread"]
    )

    # Baseline: next week's mortgage rate equals current mortgage rate.
    backtest["baseline_next_mortgage_rate"] = backtest["MORTGAGE30US"]

    # Optional component baselines
    backtest["baseline_dgs10_plus_model_spread"] = (
        backtest["DGS10"] + backtest["pred_next_spread"]
    )
    backtest["model_dgs10_plus_baseline_spread"] = (
        backtest["pred_next_dgs10"] + backtest["mortgage_spread"]
    )

    test = backtest[backtest["split"] == "test"].copy()

    summary = pd.DataFrame(
        [
            metric_row(
                "combined_component_model",
                test["actual_next_mortgage_rate"],
                test["pred_next_mortgage_rate"],
            ),
            metric_row(
                "no_change_mortgage_baseline",
                test["actual_next_mortgage_rate"],
                test["baseline_next_mortgage_rate"],
            ),
            metric_row(
                "baseline_dgs10_plus_model_spread",
                test["actual_next_mortgage_rate"],
                test["baseline_dgs10_plus_model_spread"],
            ),
            metric_row(
                "model_dgs10_plus_baseline_spread",
                test["actual_next_mortgage_rate"],
                test["model_dgs10_plus_baseline_spread"],
            ),
        ]
    ).sort_values("rmse")

    backtest.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nCombined mortgage forecast backtest:")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)

    print("\nRecent forecast rows:")
    print(
        test[
            [
                "week_date",
                "MORTGAGE30US",
                "actual_next_mortgage_rate",
                "pred_next_mortgage_rate",
                "baseline_next_mortgage_rate",
                "pred_dgs10_chg_1w",
                "pred_spread_chg_1w",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
