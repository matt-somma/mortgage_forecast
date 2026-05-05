import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from project_paths import MODEL_RESULTS_DIR, PROCESSED_DATA_DIR


SPINE_PATH = PROCESSED_DATA_DIR / "weekly_macro_spine.csv"
OUTPUT_PATH = MODEL_RESULTS_DIR / "walk_forward_arimax_spread.csv"


SPREAD_EXOG = [
    "mortgage_spread",
    "mortgage_spread_chg_1w",
    "mortgage_spread_gap_vs_5y_avg",
    "dgs10_chg_1w",
    "nfci_chg_1w",
    "vixcls_chg_1w",
    "credit_spread_baa_aaa",
]


def main():
    df = pd.read_csv(SPINE_PATH, parse_dates=["week_date"]).sort_values("week_date")

    # Targets
    df["target_spread_chg_1w"] = df["mortgage_spread"].shift(-1) - df["mortgage_spread"]
    df["target_mortgage"] = df["MORTGAGE30US"].shift(-1)

    df = df.dropna(subset=SPREAD_EXOG + ["target_spread_chg_1w", "target_mortgage"])

    start_idx = int(len(df) * 0.7)

    results = []

    for i in range(start_idx, len(df) - 1):
        train_df = df.iloc[:i]
        test_row = df.iloc[i]

        y_train = train_df["target_spread_chg_1w"]
        X_train = train_df[SPREAD_EXOG]

        try:
            model = SARIMAX(
                y_train,
                exog=X_train,
                order=(1, 1, 1),   # start simple
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            model_fit = model.fit(disp=False)

            X_test = test_row[SPREAD_EXOG].to_frame().T

            pred_spread = model_fit.forecast(
                steps=1,
                exog=X_test
            )[0]

        except Exception:
            # fallback if model fails
            pred_spread = 0.0

        # reconstruct mortgage rate using actual DGS10
        pred_rate = (
            test_row["DGS10"] +
            test_row["mortgage_spread"] +
            pred_spread
        )

        baseline_rate = test_row["MORTGAGE30US"]

        results.append({
            "week_date": test_row["week_date"],
            "actual": test_row["target_mortgage"],
            "predicted": pred_rate,
            "baseline": baseline_rate,
        })

    results_df = pd.DataFrame(results)

    rmse_model = root_mean_squared_error(results_df["actual"], results_df["predicted"])
    rmse_baseline = root_mean_squared_error(results_df["actual"], results_df["baseline"])

    mae_model = mean_absolute_error(results_df["actual"], results_df["predicted"])
    mae_baseline = mean_absolute_error(results_df["actual"], results_df["baseline"])

    print("\n=== ARIMAX SPREAD WALK-FORWARD ===")
    print("Model RMSE:", rmse_model)
    print("Baseline RMSE:", rmse_baseline)
    print("Model MAE:", mae_model)
    print("Baseline MAE:", mae_baseline)

    results_df.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
