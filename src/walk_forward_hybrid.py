import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from project_paths import MODEL_RESULTS_DIR, PROCESSED_DATA_DIR


SPINE_PATH = PROCESSED_DATA_DIR / "weekly_macro_spine.csv"
OUTPUT_PATH = MODEL_RESULTS_DIR / "walk_forward_hybrid.csv"


DGS10_ALPHA = 1000
SPREAD_ALPHA = 500


DGS10_FEATURES = [
    "DGS10","DGS2","DGS3MO","DGS5","DGS30",
    "DFII10","T10YIE",
    "yield_curve_10y_2y","yield_curve_10y_3mo","yield_curve_30y_10y",
    "dgs10_chg_1w","dgs10_chg_4w",
    "dgs2_chg_1w","dgs5_chg_1w","dgs30_chg_1w",
    "dgs10_roll_std_4w","dgs10_roll_std_12w",
    "VIXCLS","vixcls_chg_1w","vixcls_roll_std_4w"
]

SPREAD_FEATURES = [
    "mortgage_spread","MORTGAGE30US","DGS10",
    "BAA","AAA","NFCI","VIXCLS",
    "credit_spread_baa_aaa","credit_spread_baa_10y",
    "mortgage_spread_chg_1w","mortgage_spread_chg_4w",
    "mortgage_spread_roll_std_4w","mortgage_spread_roll_std_12w",
    "mortgage_spread_gap_vs_5y_avg",
    "dgs10_chg_1w","dgs10_chg_4w",
    "vixcls_chg_1w","nfci_chg_1w"
]


def build_ridge(train_df):
    dgs_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=DGS10_ALPHA))
    ])

    spread_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=SPREAD_ALPHA))
    ])

    dgs_model.fit(train_df[DGS10_FEATURES], train_df["target_dgs10_chg_1w"])
    spread_model.fit(train_df[SPREAD_FEATURES], train_df["target_spread_chg_1w"])

    return dgs_model, spread_model


def build_residual_models(train_df, dgs_model, spread_model):
    # Predict training
    pred_dgs = dgs_model.predict(train_df[DGS10_FEATURES])
    pred_spread = spread_model.predict(train_df[SPREAD_FEATURES])

    # Residuals
    res_dgs = train_df["target_dgs10_chg_1w"] - pred_dgs
    res_spread = train_df["target_spread_chg_1w"] - pred_spread

    xgb_dgs = XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_spread = XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_dgs.fit(train_df[DGS10_FEATURES].astype(float), res_dgs.astype(float))
    xgb_spread.fit(train_df[SPREAD_FEATURES].astype(float), res_spread.astype(float))

    return xgb_dgs, xgb_spread


def main():
    df = pd.read_csv(SPINE_PATH, parse_dates=["week_date"]).sort_values("week_date")

    df["target_dgs10_chg_1w"] = df["DGS10"].shift(-1) - df["DGS10"]
    df["target_spread_chg_1w"] = df["mortgage_spread"].shift(-1) - df["mortgage_spread"]
    df["target_mortgage"] = df["MORTGAGE30US"].shift(-1)

    df = df.dropna()

    start_idx = int(len(df) * 0.7)

    results = []

    for i in range(start_idx, len(df) - 1):
        train_df = df.iloc[:i]
        test_row = df.iloc[i]

        # Step 1: Ridge
        dgs_model, spread_model = build_ridge(train_df)

        # Step 2: Residual models
        xgb_dgs, xgb_spread = build_residual_models(train_df, dgs_model, spread_model)

        x_dgs = test_row[DGS10_FEATURES].to_frame().T.astype(float)
        x_spread = test_row[SPREAD_FEATURES].to_frame().T.astype(float)

        ridge_dgs = dgs_model.predict(x_dgs)[0]
        ridge_spread = spread_model.predict(x_spread)[0]

        res_dgs = xgb_dgs.predict(x_dgs)[0]
        res_spread = xgb_spread.predict(x_spread)[0]

        # Combine
        pred_dgs = ridge_dgs + res_dgs
        pred_spread = ridge_spread + res_spread

        pred_rate = (
            test_row["DGS10"] + pred_dgs +
            test_row["mortgage_spread"] + pred_spread
        )

        results.append({
            "week_date": test_row["week_date"],
            "actual": test_row["target_mortgage"],
            "predicted": pred_rate,
            "baseline": test_row["MORTGAGE30US"]
        })

    results_df = pd.DataFrame(results)

    rmse_model = root_mean_squared_error(results_df["actual"], results_df["predicted"])
    rmse_baseline = root_mean_squared_error(results_df["actual"], results_df["baseline"])

    mae_model = mean_absolute_error(results_df["actual"], results_df["predicted"])
    mae_baseline = mean_absolute_error(results_df["actual"], results_df["baseline"])

    print("\n=== HYBRID MODEL (RIDGE + XGB RESIDUAL) ===")
    print("Model RMSE:", rmse_model)
    print("Baseline RMSE:", rmse_baseline)
    print("Model MAE:", mae_model)
    print("Baseline MAE:", mae_baseline)

    results_df.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
