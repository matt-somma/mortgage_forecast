import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from project_paths import PROCESSED_DATA_DIR


SPINE_PATH = PROCESSED_DATA_DIR / "weekly_macro_spine.csv"


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


DGS10_ALPHA = 1000
SPREAD_ALPHA = 500


def run_horizon(df, horizon):
    print(f"\n=== HORIZON: {horizon}-WEEK ===")

    # Targets
    df[f"target_dgs10_{horizon}w"] = df["DGS10"].shift(-horizon) - df["DGS10"]
    df[f"target_spread_{horizon}w"] = df["mortgage_spread"].shift(-horizon) - df["mortgage_spread"]
    df[f"target_mortgage_{horizon}w"] = df["MORTGAGE30US"].shift(-horizon)

    work_df = df.dropna().copy()

    start_idx = int(len(work_df) * 0.7)

    preds = []
    actuals = []
    baseline = []

    for i in range(start_idx, len(work_df) - horizon):
        train_df = work_df.iloc[:i]
        row = work_df.iloc[i]

        # Ridge models
        dgs_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=DGS10_ALPHA))
        ])

        spread_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=SPREAD_ALPHA))
        ])

        dgs_model.fit(
            train_df[DGS10_FEATURES],
            train_df[f"target_dgs10_{horizon}w"]
        )

        spread_model.fit(
            train_df[SPREAD_FEATURES],
            train_df[f"target_spread_{horizon}w"]
        )

        x_dgs = row[DGS10_FEATURES].to_frame().T
        x_spread = row[SPREAD_FEATURES].to_frame().T

        pred_dgs = dgs_model.predict(x_dgs)[0]
        pred_spread = spread_model.predict(x_spread)[0]

        pred_rate = (
            row["DGS10"] + pred_dgs +
            row["mortgage_spread"] + pred_spread
        )

        preds.append(pred_rate)
        actuals.append(row[f"target_mortgage_{horizon}w"])
        baseline.append(row["MORTGAGE30US"])

    rmse_model = root_mean_squared_error(actuals, preds)
    rmse_base = root_mean_squared_error(actuals, baseline)

    mae_model = mean_absolute_error(actuals, preds)
    mae_base = mean_absolute_error(actuals, baseline)

    print("Model RMSE:", rmse_model)
    print("Baseline RMSE:", rmse_base)
    print("Model MAE:", mae_model)
    print("Baseline MAE:", mae_base)


def main():
    df = pd.read_csv(SPINE_PATH, parse_dates=["week_date"]).sort_values("week_date")

    run_horizon(df.copy(), 4)
    run_horizon(df.copy(), 12)


if __name__ == "__main__":
    main()
