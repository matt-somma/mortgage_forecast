import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.project_paths import CHARTS_DIR, MODEL_RESULTS_DIR, PROCESSED_DATA_DIR


SPINE_PATH = PROCESSED_DATA_DIR / "weekly_macro_spine.csv"
OUTPUT_DIR = MODEL_RESULTS_DIR
CHART_DIR = CHARTS_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

PATHS_OUTPUT = OUTPUT_DIR / "monte_carlo_12mo_paths.csv"
SCENARIO_SUMMARY_OUTPUT = OUTPUT_DIR / "monte_carlo_12mo_scenario_summary.csv"
WEIGHTED_PATHS_BY_VIEW_OUTPUT = OUTPUT_DIR / "monte_carlo_12mo_weighted_paths_by_view.csv"
WEIGHTED_SUMMARY_BY_VIEW_OUTPUT = OUTPUT_DIR / "monte_carlo_12mo_weighted_summary_by_view.csv"
FINAL_WEEK_BY_VIEW_OUTPUT = OUTPUT_DIR / "monte_carlo_12mo_final_week_by_view.csv"

CURRENT_VIEW_CHART_PATH = CHART_DIR / "monte_carlo_12mo_fan_chart.png"
VIEW_GRID_CHART_PATH = CHART_DIR / "monte_carlo_12mo_fan_chart_by_view_grid.png"

N_SIMS = 5000
HORIZON_WEEKS = 52
RANDOM_SEED = 42

DGS10_ALPHA = 1000
SPREAD_ALPHA = 500

MIN_DGS10 = 0.25
MIN_SPREAD = 0.25
MAX_SPREAD = 4.50

DGS10_MEAN_REVERSION = 0.015
SPREAD_MEAN_REVERSION = 0.040


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
        "dgs10_shock_mean": 0.000,
        "spread_shock_mean": 0.000,
        "vol_multiplier": 1.00,
        "long_run_dgs10": 4.25,
        "spread_anchor_multiplier": 1.00,
    },
    "higher_for_longer": {
        "dgs10_shock_mean": 0.006,
        "spread_shock_mean": 0.001,
        "vol_multiplier": 1.10,
        "long_run_dgs10": 4.85,
        "spread_anchor_multiplier": 1.05,
    },
    "soft_landing": {
        "dgs10_shock_mean": -0.003,
        "spread_shock_mean": -0.002,
        "vol_multiplier": 0.85,
        "long_run_dgs10": 3.90,
        "spread_anchor_multiplier": 0.95,
    },
    "recession_easing": {
        "dgs10_shock_mean": -0.010,
        "spread_shock_mean": 0.004,
        "vol_multiplier": 1.25,
        "long_run_dgs10": 3.25,
        "spread_anchor_multiplier": 1.10,
    },
    "financial_stress": {
        "dgs10_shock_mean": -0.002,
        "spread_shock_mean": 0.012,
        "vol_multiplier": 1.60,
        "long_run_dgs10": 3.75,
        "spread_anchor_multiplier": 1.30,
    },
}


ECONOMIC_VIEWS = {
    "current_view": {
        "base_case": 0.40,
        "higher_for_longer": 0.25,
        "soft_landing": 0.20,
        "recession_easing": 0.10,
        "financial_stress": 0.05,
    },
    "higher_for_longer_view": {
        "base_case": 0.25,
        "higher_for_longer": 0.45,
        "soft_landing": 0.15,
        "recession_easing": 0.10,
        "financial_stress": 0.05,
    },
    "soft_landing_view": {
        "base_case": 0.35,
        "higher_for_longer": 0.15,
        "soft_landing": 0.35,
        "recession_easing": 0.10,
        "financial_stress": 0.05,
    },
    "recession_view": {
        "base_case": 0.25,
        "higher_for_longer": 0.15,
        "soft_landing": 0.15,
        "recession_easing": 0.35,
        "financial_stress": 0.10,
    },
    "stress_view": {
        "base_case": 0.20,
        "higher_for_longer": 0.20,
        "soft_landing": 0.10,
        "recession_easing": 0.15,
        "financial_stress": 0.35,
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

    required = DGS10_FEATURES + SPREAD_FEATURES + [
        "target_dgs10_chg_1w",
        "target_spread_chg_1w",
        "mortgage_spread_roll_mean_5y",
    ]

    return df.dropna(subset=required).copy()


def update_state(
    state: pd.Series,
    dgs10: float,
    spread: float,
    mortgage_rate: float,
    dgs_chg: float,
    spread_chg: float,
) -> pd.Series:
    state = state.copy()

    state["DGS10"] = dgs10
    state["mortgage_spread"] = spread
    state["MORTGAGE30US"] = mortgage_rate

    state["dgs10_chg_1w"] = dgs_chg
    state["mortgage_spread_chg_1w"] = spread_chg

    state["dgs10_chg_4w"] = 0.75 * state["dgs10_chg_4w"] + dgs_chg
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

    return state


def simulate_scenario(
    df: pd.DataFrame,
    scenario_name: str,
    scenario: dict,
    dgs_model,
    spread_model,
    dgs_sigma: float,
    spread_sigma: float,
) -> pd.DataFrame:
    scenario_seed = RANDOM_SEED + abs(hash(scenario_name)) % 10000
    rng = np.random.default_rng(scenario_seed)

    latest = df.iloc[-1].copy()

    current_dgs10 = float(latest["DGS10"])
    current_spread = float(latest["mortgage_spread"])

    spread_anchor = (
        float(latest["mortgage_spread_roll_mean_5y"])
        * scenario["spread_anchor_multiplier"]
    )

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

            dgs_anchor_pull = DGS10_MEAN_REVERSION * (
                scenario["long_run_dgs10"] - dgs10
            )

            spread_anchor_pull = SPREAD_MEAN_REVERSION * (
                spread_anchor - spread
            )

            dgs_shock = rng.normal(
                scenario["dgs10_shock_mean"],
                dgs_sigma * scenario["vol_multiplier"],
            )

            spread_shock = rng.normal(
                scenario["spread_shock_mean"],
                spread_sigma * scenario["vol_multiplier"],
            )

            dgs_chg = pred_dgs_chg + dgs_anchor_pull + dgs_shock
            spread_chg = pred_spread_chg + spread_anchor_pull + spread_shock

            dgs10 = max(dgs10 + dgs_chg, MIN_DGS10)
            spread = min(max(spread + spread_chg, MIN_SPREAD), MAX_SPREAD)

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

            state = update_state(
                state=state,
                dgs10=dgs10,
                spread=spread,
                mortgage_rate=mortgage_rate,
                dgs_chg=dgs_chg,
                spread_chg=spread_chg,
            )

    return pd.DataFrame(records)


def summarize_scenario_paths(paths: pd.DataFrame) -> pd.DataFrame:
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


def build_weighted_outputs_for_views(paths: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_weighted_paths = []
    all_weighted_summaries = []

    for view_name, weights in ECONOMIC_VIEWS.items():
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError(f"Weights for {view_name} do not sum to 1.")

        view_samples = []

        for scenario_name, weight in weights.items():
            scenario_paths = paths[paths["scenario"] == scenario_name].copy()

            if scenario_paths.empty:
                raise ValueError(f"No paths found for scenario: {scenario_name}")

            sample_n = max(1, int(N_SIMS * weight))

            sampled_ids = (
                scenario_paths[["sim_id"]]
                .drop_duplicates()
                .sample(n=sample_n, replace=True, random_state=RANDOM_SEED)
            )

            sampled = scenario_paths.merge(sampled_ids, on="sim_id", how="inner")
            sampled["economic_view"] = view_name
            sampled["view_weight"] = weight

            view_samples.append(sampled)

        view_paths = pd.concat(view_samples, ignore_index=True)
        all_weighted_paths.append(view_paths)

        summary = (
            view_paths.groupby("week")["mortgage_rate"]
            .quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
            .unstack()
            .reset_index()
        )

        summary.columns = [
            "week",
            "p05",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
        ]

        summary["economic_view"] = view_name
        all_weighted_summaries.append(summary)

    return (
        pd.concat(all_weighted_paths, ignore_index=True),
        pd.concat(all_weighted_summaries, ignore_index=True),
    )


def make_single_fan_chart(
    weighted_summary_by_view: pd.DataFrame,
    latest_rate: float,
    view_name: str = "current_view",
) -> None:
    data = weighted_summary_by_view[
        weighted_summary_by_view["economic_view"] == view_name
    ].copy()

    plt.figure(figsize=(12, 7))

    plt.plot(
        [0] + data["week"].tolist(),
        [latest_rate] + data["p50"].tolist(),
        label="Median Forecast",
        linewidth=2,
    )

    plt.fill_between(
        data["week"],
        data["p25"],
        data["p75"],
        alpha=0.30,
        label="25th-75th Percentile",
    )

    plt.fill_between(
        data["week"],
        data["p10"],
        data["p90"],
        alpha=0.18,
        label="10th-90th Percentile",
    )

    plt.fill_between(
        data["week"],
        data["p05"],
        data["p95"],
        alpha=0.10,
        label="5th-95th Percentile",
    )

    plt.axhline(
        latest_rate,
        linestyle="--",
        linewidth=1,
        label="Current Mortgage Rate",
    )

    title = view_name.replace("_", " ").title()
    plt.title(f"12-Month Monte Carlo Mortgage Rate Forecast: {title}")
    plt.xlabel("Weeks Ahead")
    plt.ylabel("30-Year Mortgage Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURRENT_VIEW_CHART_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved current-view fan chart: {CURRENT_VIEW_CHART_PATH}")


def make_fan_chart_grid(
    weighted_summary_by_view: pd.DataFrame,
    latest_rate: float,
) -> None:
    view_order = [
        "current_view",
        "higher_for_longer_view",
        "soft_landing_view",
        "recession_view",
        "stress_view",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, view_name in zip(axes, view_order):
        data = weighted_summary_by_view[
            weighted_summary_by_view["economic_view"] == view_name
        ].copy()

        title = view_name.replace("_", " ").title()

        ax.plot(
            [0] + data["week"].tolist(),
            [latest_rate] + data["p50"].tolist(),
            linewidth=2,
            label="Median",
        )

        ax.fill_between(
            data["week"],
            data["p25"],
            data["p75"],
            alpha=0.30,
            label="p25-p75",
        )

        ax.fill_between(
            data["week"],
            data["p10"],
            data["p90"],
            alpha=0.18,
            label="p10-p90",
        )

        ax.fill_between(
            data["week"],
            data["p05"],
            data["p95"],
            alpha=0.10,
            label="p05-p95",
        )

        ax.axhline(latest_rate, linestyle="--", linewidth=1)

        final = data[data["week"] == HORIZON_WEEKS].iloc[0]

        ax.set_title(
            f"{title}\n"
            f"52w median: {final['p50']:.2f}% | "
            f"p10-p90: {final['p10']:.2f}%–{final['p90']:.2f}%"
        )

        ax.set_xlabel("Weeks Ahead")
        ax.set_ylabel("Mortgage Rate (%)")
        ax.grid(True, alpha=0.25)

    # Hide unused 6th subplot.
    axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)

    fig.suptitle(
        "12-Month Monte Carlo Mortgage Rate Forecast by Economic View",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(VIEW_GRID_CHART_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved fan chart grid: {VIEW_GRID_CHART_PATH}")


def main() -> None:
    df = prepare_data()

    dgs_model = fit_model(
        df,
        DGS10_FEATURES,
        "target_dgs10_chg_1w",
        DGS10_ALPHA,
    )

    spread_model = fit_model(
        df,
        SPREAD_FEATURES,
        "target_spread_chg_1w",
        SPREAD_ALPHA,
    )

    dgs_resid = df["target_dgs10_chg_1w"] - dgs_model.predict(
        df[DGS10_FEATURES].astype(float)
    )

    spread_resid = df["target_spread_chg_1w"] - spread_model.predict(
        df[SPREAD_FEATURES].astype(float)
    )

    dgs_sigma = float(dgs_resid.std())
    spread_sigma = float(spread_resid.std())

    scenario_path_frames = []

    for scenario_name, scenario in SCENARIOS.items():
        print(f"Simulating {scenario_name}...")

        scenario_path_frames.append(
            simulate_scenario(
                df=df,
                scenario_name=scenario_name,
                scenario=scenario,
                dgs_model=dgs_model,
                spread_model=spread_model,
                dgs_sigma=dgs_sigma,
                spread_sigma=spread_sigma,
            )
        )

    paths_df = pd.concat(scenario_path_frames, ignore_index=True)
    scenario_summary = summarize_scenario_paths(paths_df)

    weighted_paths_by_view, weighted_summary_by_view = build_weighted_outputs_for_views(
        paths_df
    )

    final_week_summary = weighted_summary_by_view[
        weighted_summary_by_view["week"] == HORIZON_WEEKS
    ].copy()

    paths_df.to_csv(PATHS_OUTPUT, index=False)
    scenario_summary.to_csv(SCENARIO_SUMMARY_OUTPUT, index=False)
    weighted_paths_by_view.to_csv(WEIGHTED_PATHS_BY_VIEW_OUTPUT, index=False)
    weighted_summary_by_view.to_csv(WEIGHTED_SUMMARY_BY_VIEW_OUTPUT, index=False)
    final_week_summary.to_csv(FINAL_WEEK_BY_VIEW_OUTPUT, index=False)

    latest_rate = float(df.iloc[-1]["MORTGAGE30US"])

    make_single_fan_chart(
        weighted_summary_by_view=weighted_summary_by_view,
        latest_rate=latest_rate,
        view_name="current_view",
    )

    make_fan_chart_grid(
        weighted_summary_by_view=weighted_summary_by_view,
        latest_rate=latest_rate,
    )

    print("\nSaved:")
    print(PATHS_OUTPUT)
    print(SCENARIO_SUMMARY_OUTPUT)
    print(WEIGHTED_PATHS_BY_VIEW_OUTPUT)
    print(WEIGHTED_SUMMARY_BY_VIEW_OUTPUT)
    print(FINAL_WEEK_BY_VIEW_OUTPUT)
    print(CURRENT_VIEW_CHART_PATH)
    print(VIEW_GRID_CHART_PATH)

    print("\nEconomic views and weights:")
    for view_name, weights in ECONOMIC_VIEWS.items():
        print(f"\n{view_name}:")
        for scenario, weight in weights.items():
            print(f"  {scenario}: {weight:.0%}")

    print("\n52-week forecast by economic view:")
    print(
        final_week_summary[
            ["economic_view", "p05", "p10", "p25", "p50", "p75", "p90", "p95"]
        ]
        .sort_values("p50")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
