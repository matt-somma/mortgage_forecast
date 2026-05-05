import pandas as pd
import matplotlib.pyplot as plt
from project_paths import CHARTS_DIR, MODEL_RESULTS_DIR, PROCESSED_DATA_DIR


PROCESSED_DIR = PROCESSED_DATA_DIR
RESULTS_DIR = MODEL_RESULTS_DIR
VISUALS_DIR = CHARTS_DIR
VISUALS_DIR.mkdir(parents=True, exist_ok=True)


SPINE_PATH = PROCESSED_DIR / "weekly_macro_spine.csv"
COMBINED_BACKTEST_PATH = RESULTS_DIR / "combined_mortgage_forecast_backtest.csv"
DIRECT_VS_DECOMP_PATH = RESULTS_DIR / "direct_vs_decomposed_summary.csv"
WALK_FORWARD_PATH = RESULTS_DIR / "walk_forward_results.csv"


def save_chart(filename: str):
    path = VISUALS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def mortgage_vs_treasury_chart(spine: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(spine["week_date"], spine["MORTGAGE30US"], label="30Y Mortgage Rate")
    plt.plot(spine["week_date"], spine["DGS10"], label="10Y Treasury Yield")
    plt.title("30-Year Mortgage Rate vs 10-Year Treasury Yield")
    plt.xlabel("Date")
    plt.ylabel("Rate (%)")
    plt.legend()
    save_chart("mortgage_rate_vs_10y_treasury.png")


def mortgage_spread_chart(spine: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(spine["week_date"], spine["mortgage_spread"], label="Mortgage Spread")
    plt.plot(
        spine["week_date"],
        spine["mortgage_spread_roll_mean_5y"],
        label="5-Year Rolling Average",
    )
    plt.title("Mortgage Spread Over Time")
    plt.xlabel("Date")
    plt.ylabel("Spread (%)")
    plt.legend()
    save_chart("mortgage_spread_over_time.png")


def actual_vs_predicted_chart(backtest: pd.DataFrame):
    test = backtest[backtest["split"] == "test"].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(
        test["week_date"],
        test["actual_next_mortgage_rate"],
        label="Actual Next Mortgage Rate",
    )
    plt.plot(
        test["week_date"],
        test["pred_next_mortgage_rate"],
        label="Predicted Next Mortgage Rate",
    )
    plt.title("Actual vs Predicted Mortgage Rate")
    plt.xlabel("Date")
    plt.ylabel("Mortgage Rate (%)")
    plt.legend()
    save_chart("actual_vs_predicted_mortgage_rate.png")


def model_comparison_chart(summary: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.bar(summary["model_name"], summary["rmse"])
    plt.title("Model Comparison by RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    save_chart("model_comparison_rmse.png")


def walk_forward_chart(walk_forward: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(walk_forward["week_date"], walk_forward["actual"], label="Actual")
    plt.plot(walk_forward["week_date"], walk_forward["predicted"], label="Predicted")
    plt.plot(walk_forward["week_date"], walk_forward["baseline"], label="No-Change Baseline")
    plt.title("Walk-Forward Backtest: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Mortgage Rate (%)")
    plt.legend()
    save_chart("walk_forward_actual_vs_predicted.png")


def multi_horizon_chart():
    data = pd.DataFrame(
        {
            "horizon": ["1-week", "4-week", "12-week"],
            "model_rmse": [0.04615445615110869, 0.1054671895387169, 0.24131905644757315],
            "baseline_rmse": [0.058766388115086846, 0.1203328716519306, 0.2879832894457593],
        }
    )

    x = range(len(data))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], data["model_rmse"], width=width, label="Model")
    plt.bar([i + width / 2 for i in x], data["baseline_rmse"], width=width, label="Baseline")
    plt.xticks(list(x), data["horizon"])
    plt.title("Multi-Horizon RMSE Comparison")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("RMSE")
    plt.legend()
    save_chart("multi_horizon_rmse_comparison.png")


def main():
    spine = pd.read_csv(SPINE_PATH, parse_dates=["week_date"])
    backtest = pd.read_csv(COMBINED_BACKTEST_PATH, parse_dates=["week_date"])
    summary = pd.read_csv(DIRECT_VS_DECOMP_PATH)
    walk_forward = pd.read_csv(WALK_FORWARD_PATH, parse_dates=["week_date"])

    mortgage_vs_treasury_chart(spine)
    mortgage_spread_chart(spine)
    actual_vs_predicted_chart(backtest)
    model_comparison_chart(summary)
    walk_forward_chart(walk_forward)
    multi_horizon_chart()

    print("\nAll charts saved to:")
    print(VISUALS_DIR)


if __name__ == "__main__":
    main()
