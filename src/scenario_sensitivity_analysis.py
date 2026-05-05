import pandas as pd
import matplotlib.pyplot as plt
from project_paths import CHARTS_DIR, MODEL_RESULTS_DIR


PATHS_PATH = MODEL_RESULTS_DIR / "monte_carlo_12mo_paths.csv"
OUTPUT_DIR = MODEL_RESULTS_DIR
CHART_DIR = CHARTS_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "scenario_weight_sensitivity_summary.csv"
CHART_PATH = CHART_DIR / "scenario_weight_sensitivity.png"


WEIGHT_SETS = {
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


def validate_weights(weights: dict) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1. Current sum: {total}")


def weighted_sample(paths: pd.DataFrame, weights: dict, sample_size: int = 5000) -> pd.DataFrame:
    validate_weights(weights)

    samples = []

    for scenario, weight in weights.items():
        scenario_paths = paths[paths["scenario"] == scenario].copy()

        if scenario_paths.empty:
            raise ValueError(f"No paths found for scenario: {scenario}")

        n = max(1, int(sample_size * weight))

        sim_ids = (
            scenario_paths[["sim_id"]]
            .drop_duplicates()
            .sample(n=n, replace=True, random_state=42)
        )

        sampled = scenario_paths.merge(sim_ids, on="sim_id", how="inner")
        samples.append(sampled)

    return pd.concat(samples, ignore_index=True)


def summarize_final_week(weighted_paths: pd.DataFrame, view_name: str) -> dict:
    final_week = weighted_paths["week"].max()
    final = weighted_paths[weighted_paths["week"] == final_week]

    q = final["mortgage_rate"].quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

    return {
        "view_name": view_name,
        "week": final_week,
        "p05": q.loc[0.05],
        "p10": q.loc[0.10],
        "p25": q.loc[0.25],
        "p50": q.loc[0.50],
        "p75": q.loc[0.75],
        "p90": q.loc[0.90],
        "p95": q.loc[0.95],
        "central_50_width": q.loc[0.75] - q.loc[0.25],
        "central_90_width": q.loc[0.95] - q.loc[0.05],
    }


def build_full_percentile_path(weighted_paths: pd.DataFrame, view_name: str) -> pd.DataFrame:
    summary = (
        weighted_paths.groupby("week")["mortgage_rate"]
        .quantile([0.10, 0.50, 0.90])
        .unstack()
        .reset_index()
    )

    summary.columns = ["week", "p10", "p50", "p90"]
    summary["view_name"] = view_name

    return summary


def make_chart(final_summary: pd.DataFrame) -> None:
    plot_df = final_summary.sort_values("p50").copy()

    x = range(len(plot_df))

    plt.figure(figsize=(12, 7))

    plt.errorbar(
        x,
        plot_df["p50"],
        yerr=[
            plot_df["p50"] - plot_df["p10"],
            plot_df["p90"] - plot_df["p50"],
        ],
        fmt="o",
        capsize=5,
        label="p10-p90 range with median",
    )

    plt.xticks(x, plot_df["view_name"], rotation=25, ha="right")
    plt.title("Scenario Weight Sensitivity: 52-Week Mortgage Rate Forecast")
    plt.xlabel("Macro Weighting View")
    plt.ylabel("Mortgage Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved chart: {CHART_PATH}")


def main():
    paths = pd.read_csv(PATHS_PATH)

    final_summaries = []
    path_summaries = []

    for view_name, weights in WEIGHT_SETS.items():
        print(f"Running sensitivity view: {view_name}")

        weighted_paths = weighted_sample(paths, weights)
        final_summaries.append(summarize_final_week(weighted_paths, view_name))
        path_summaries.append(build_full_percentile_path(weighted_paths, view_name))

    final_summary = pd.DataFrame(final_summaries)
    path_summary = pd.concat(path_summaries, ignore_index=True)

    final_summary.to_csv(OUTPUT_PATH, index=False)

    path_summary.to_csv(
        OUTPUT_DIR / "scenario_weight_sensitivity_paths.csv",
        index=False,
    )

    make_chart(final_summary)

    print("\n52-week sensitivity summary:")
    print(final_summary.sort_values("p50").to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(OUTPUT_DIR / "scenario_weight_sensitivity_paths.csv")


if __name__ == "__main__":
    main()
