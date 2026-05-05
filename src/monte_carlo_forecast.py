import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from project_paths import CHARTS_DIR, MODEL_RESULTS_DIR, PROCESSED_DATA_DIR


SPINE_PATH = PROCESSED_DATA_DIR / "weekly_macro_spine.csv"
OUTPUT_DIR = MODEL_RESULTS_DIR
CHART_DIR = CHARTS_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

SIM_OUTPUT_PATH = OUTPUT_DIR / "monte_carlo_mortgage_rate_paths.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "monte_carlo_mortgage_rate_summary.csv"
CHART_PATH = CHART_DIR / "monte_carlo_mortgage_rate_forecast.png"

N_SIMS = 5000
HORIZON_WEEKS = 52
RANDOM_SEED = 42

DGS10_ALPHA = 1000
SPREAD_ALPHA = 500


DGS10_FEATURES = [
    "DGS10", "DGS2", "DGS3MO", "DGS5", "DGS30",
    "DFII10", "T10YIE",
    "yield_curve_10y_2y", "yield_curve_10y_3mo", "yield_curve_30y_10y",
    "dgs10_chg_1w", "dgs10_chg_4w",
    "dgs2_chg_1w", "dgs5_chg_1w", "dgs30_chg_1w",
    "dgs10_roll_std_4w", "dgs10_roll_std_12w",
    "VIXCLS", "vixcls_chg_1w", "vixcls_roll_std_4w",
]

SPREAD_FEATURES = [
    "mortgage_spread", "MORTGAGE30US", "DGS10",
    "BAA", "AAA", "NFCI", "VIXCLS",
    "credit_spread_baa_aaa", "credit_spread_baa_10y",
    "mortgage_spread_chg_1w", "mortgage_spread_chg_4w",
    "mortgage_spread_roll_std_4w", "mortgage_spread_roll_std_12w",
    "mortgage_spread_gap_vs_5y_avg",
    "dgs10_chg_1w", "dgs10_chg_4w",
    "vixcls_chg_1w", "nfci_chg_1w",
]


SCENARIOS = {
    "base_case": {
        "dgs10_shock_mean": 0.00,
        "spread_shock_mean": 0.00,
        "vol_multiplier": 1.00,
    },
    "rate_shock_up": {
        "dgs10_shock_mean": 0.02,
        "spread_shock_mean": 0.00,
        "vol_multiplier": 1.10,
    },
    "rate_shock_down": {
        "dgs10_shock_mean": -0.02,
        "spread_shock_mean": 0.00,
        "vol_multiplier": 1.10,
    },
    "stress_widening": {
        "dgs10_shock_mean": 0.00,
        "spread_shock_mean": 0.02,
        "vol_multiplier": 1.35,
    },
    "easing_narrowing": {
        "dgs10_shock_mean": -0.01,
        "spread_shock_mean": -0.01,
        "vol_multiplier": 0.90,
    },
}


def fit_model(df: pd.DataFrame, features: list[str], target: str, alpha: float):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )

    model.fit(df[features].astype(float), df[target].astype(float))
    return model


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(SPINE_PATH, parse_dates=["week_date"]).sort_values("week_date")

    df["target_dgs10_chg_1w"] = df["DGS10"].shift(-1) - df["DGS10"]
    df["target_spread_chg_1w"] = df["mortgage_spread"].shift(-1) - df["mortgage_spread"]

    df = df.dropna(subset=DGS10_FEATURES + SPREAD_FEATURES + [
        "target_dgs10_chg_1w",
        "target_spread_chg_1w",
    ])

    return df


def simulate_scenario(df: pd.DataFrame, scenario_name: str, scenario: dict) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    dgs_model = fit_model(df, DGS10_FEATURES, "target_dgs10_chg_1w", DGS10_ALPHA)
    spread_model = fit_model(df, SPREAD_FEATURES, "target_spread_chg_1w", SPREAD_ALPHA)

    dgs_resid = df["target_dgs10_chg_1w"] - dgs_model.predict(df[DGS10_FEATURES].astype(float))
    spread_resid = df["target_spread_chg_1w"] - spread_model.predict(df[SPREAD_FEATURES].astype(float))

    dgs_sigma = dgs_resid.std() * scenario["vol_multiplier"]
    spread_sigma = spread_resid.std() * scenario["vol_multiplier"]

    latest = df.iloc[-1].copy()

    current_dgs10 = latest["DGS10"]
    current_spread = latest["mortgage_spread"]

    records = []

    for sim_id in range(N_SIMS):
        dgs10 = current_dgs10
        spread = current_spread

        state = latest.copy()

        for week in range(1, HORIZON_WEEKS + 1):
            x_dgs = state[DGS10_FEATURES].to_frame().T.astype(float)
            x_spread = state[SPREAD_FEATURES].to_frame().T.astype(float)

            pred_dgs_chg = dgs_model.predict(x_dgs)[0]
            pred_spread_chg = spread_model.predict(x_spread)[0]

            dgs_shock = rng.normal(
                scenario["dgs10_shock_mean"],
                dgs_sigma,
            )

            spread_shock = rng.normal(
                scenario["spread_shock_mean"],
                spread_sigma,
            )

            dgs_chg = pred_dgs_chg + dgs_shock
            spread_chg = pred_spread_chg + spread_shock

            dgs10 = max(dgs10 + dgs_chg, 0.0)
            spread = max(spread + spread_chg, 0.25)

            mortgage_rate = dgs10 + spread

            records.append(
                {
                    "scenario": scenario_name,
                    "sim_id": sim_id,
                    "week": week,
                    "dgs10": dgs10,
                    "mortgage_spread": spread,
                    "mortgage_rate": mortgage_rate,
                }
            )

            # Simple recursive state update.
            state["DGS10"] = dgs10
            state["mortgage_spread"] = spread
            state["MORTGAGE30US"] = mortgage_rate

            state["dgs10_chg_1w"] = dgs_chg
            state["mortgage_spread_chg_1w"] = spread_chg

            state["dgs10_chg_4w"] = (
                0.75 * state["dgs10_chg_4w"] + dgs_chg
            )

            state["mortgage_spread_chg_4w"] = (
                0.75 * state["mortgage_spread_chg_4w"] + spread_chg
            )

            state["yield_curve_10y_2y"] = state["DGS10"] - state["DGS2"]
            state["yield_curve_10y_3mo"] = state["DGS10"] - state["DGS3MO"]
            state["yield_curve_30y_10y"] = state["DGS30"] - state["DGS10"]

            state["credit_spread_baa_10y"] = state["BAA"] - state["DGS10"]
            state["mortgage_spread_gap_vs_5y_avg"] = (
                state["mortgage_spread"] - state["mortgage_spread_roll_mean_5y"]
            )

    return pd.DataFrame(records)


def build_summary(paths: pd.DataFrame) -> pd.DataFrame:
    summary = (
        paths.groupby(["scenario", "week"])["mortgage_rate"]
        .quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        .unstack()
        .reset_index()
    )

    summary.columns = [
        "scenario",
        "week",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
    ]

    return summary


def make_chart(summary: pd.DataFrame):
    base = summary[summary["scenario"] == "base_case"].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(base["week"], base["p50"], label="Median Forecast")
    plt.fill_between(base["week"], base["p25"], base["p75"], alpha=0.25, label="25th-75th Percentile")
    plt.fill_between(base["week"], base["p10"], base["p90"], alpha=0.15, label="10th-90th Percentile")

    plt.title("Monte Carlo Mortgage Rate Forecast Range")
    plt.xlabel("Weeks Ahead")
    plt.ylabel("Mortgage Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved chart: {CHART_PATH}")


def main():
    df = prepare_data()

    all_paths = []

    for scenario_name, scenario in SCENARIOS.items():
        print(f"Simulating scenario: {scenario_name}")
        scenario_paths = simulate_scenario(df, scenario_name, scenario)
        all_paths.append(scenario_paths)

    paths = pd.concat(all_paths, ignore_index=True)
    summary = build_summary(paths)

    paths.to_csv(SIM_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    make_chart(summary)

    print("\nSaved:")
    print(SIM_OUTPUT_PATH)
    print(SUMMARY_OUTPUT_PATH)

    print("\nBase case final-week forecast range:")
    print(
        summary[
            (summary["scenario"] == "base_case")
            & (summary["week"] == HORIZON_WEEKS)
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
