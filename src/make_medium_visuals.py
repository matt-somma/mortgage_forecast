import pandas as pd
import matplotlib.pyplot as plt
from project_paths import MEDIUM_VISUALS_DIR, MODEL_RESULTS_DIR, PROCESSED_DATA_DIR


PROCESSED_DIR = PROCESSED_DATA_DIR
RESULTS_DIR = MODEL_RESULTS_DIR
MEDIUM_DIR = MEDIUM_VISUALS_DIR

MEDIUM_DIR.mkdir(parents=True, exist_ok=True)

SPINE_PATH = PROCESSED_DIR / "weekly_macro_spine.csv"
WALK_FORWARD_PATH = RESULTS_DIR / "walk_forward_results.csv"
DIRECT_VS_DECOMP_PATH = RESULTS_DIR / "direct_vs_decomposed_summary.csv"
MONTE_CARLO_CURRENT_PATH = RESULTS_DIR / "monte_carlo_12mo_weighted_summary_by_view.csv"
MONTE_CARLO_FINAL_VIEW_PATH = RESULTS_DIR / "monte_carlo_12mo_final_week_by_view.csv"


def save_chart(filename: str) -> None:
    path = MEDIUM_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def apply_article_style(title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title, fontsize=16, pad=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.25)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)


def chart_decomposition(spine: pd.DataFrame) -> None:
    df = spine[spine["week_date"] >= "2000-01-01"].copy()

    plt.figure(figsize=(13, 7))
    plt.plot(df["week_date"], df["MORTGAGE30US"], linewidth=2.2, label="30Y Mortgage Rate")
    plt.plot(df["week_date"], df["DGS10"], linewidth=2.0, label="10Y Treasury Yield")
    plt.plot(df["week_date"], df["mortgage_spread"], linewidth=1.8, label="Mortgage Spread")

    apply_article_style(
        title="Mortgage Rates Are Driven by Treasury Yields and Spreads",
        xlabel="Date",
        ylabel="Rate / Spread (%)",
    )

    plt.legend(frameon=False, fontsize=11)
    save_chart("01_decomposition_mortgage_treasury_spread.png")


def chart_spread_mean_reversion(spine: pd.DataFrame) -> None:
    df = spine[spine["week_date"] >= "2000-01-01"].copy()

    plt.figure(figsize=(13, 7))
    plt.plot(df["week_date"], df["mortgage_spread"], linewidth=2.1, label="Mortgage Spread")
    plt.plot(
        df["week_date"],
        df["mortgage_spread_roll_mean_5y"],
        linewidth=2.1,
        linestyle="--",
        label="5-Year Rolling Average",
    )

    apply_article_style(
        title="Mortgage Spreads Are Mean-Reverting but Stress-Sensitive",
        xlabel="Date",
        ylabel="Spread (%)",
    )

    plt.legend(frameon=False, fontsize=11)
    save_chart("02_mortgage_spread_mean_reversion.png")


def chart_walk_forward(walk_forward: pd.DataFrame) -> None:
    df = walk_forward.copy()

    plt.figure(figsize=(13, 7))
    plt.plot(df["week_date"], df["actual"], linewidth=2.2, label="Actual")
    plt.plot(df["week_date"], df["predicted"], linewidth=2.0, label="Model Forecast")
    plt.plot(df["week_date"], df["baseline"], linewidth=1.8, linestyle="--", label="No-Change Baseline")

    apply_article_style(
        title="Walk-Forward Backtest: Forecast vs Actual Mortgage Rates",
        xlabel="Date",
        ylabel="30Y Mortgage Rate (%)",
    )

    plt.legend(frameon=False, fontsize=11)
    save_chart("03_walk_forward_forecast_vs_actual.png")


def chart_model_comparison(summary: pd.DataFrame) -> None:
    df = summary.copy()

    label_map = {
        "decomposed_10y_plus_spread_model": "Decomposed Model",
        "direct_mortgage_model": "Direct Model",
        "no_change_baseline": "No-Change Baseline",
    }

    df["label"] = df["model_name"].map(label_map).fillna(df["model_name"])
    df = df.sort_values("rmse", ascending=False)

    plt.figure(figsize=(11, 6.5))
    plt.barh(df["label"], df["rmse"])

    apply_article_style(
        title="Decomposition Outperformed Direct Mortgage-Rate Forecasting",
        xlabel="RMSE",
        ylabel="Model",
    )

    for i, value in enumerate(df["rmse"]):
        plt.text(value + 0.002, i, f"{value:.3f}", va="center", fontsize=10)

    save_chart("04_model_comparison_rmse.png")


def chart_monte_carlo_current_view(mc_summary: pd.DataFrame) -> None:
    df = mc_summary[mc_summary["economic_view"] == "current_view"].copy()

    plt.figure(figsize=(13, 7))

    plt.plot(df["week"], df["p50"], linewidth=2.4, label="Median Forecast")
    plt.fill_between(df["week"], df["p25"], df["p75"], alpha=0.30, label="25th-75th Percentile")
    plt.fill_between(df["week"], df["p10"], df["p90"], alpha=0.18, label="10th-90th Percentile")
    plt.fill_between(df["week"], df["p05"], df["p95"], alpha=0.10, label="5th-95th Percentile")

    apply_article_style(
        title="12-Month Mortgage Rate Forecast: Probability Range",
        xlabel="Weeks Ahead",
        ylabel="30Y Mortgage Rate (%)",
    )

    plt.legend(frameon=False, fontsize=11)
    save_chart("05_monte_carlo_current_view_fan_chart.png")


def chart_economic_view_sensitivity(final_view: pd.DataFrame) -> None:
    df = final_view.copy()

    order = [
        "recession_view",
        "soft_landing_view",
        "current_view",
        "stress_view",
        "higher_for_longer_view",
    ]

    labels = {
        "recession_view": "Recession",
        "soft_landing_view": "Soft Landing",
        "current_view": "Current View",
        "stress_view": "Stress",
        "higher_for_longer_view": "Higher-for-Longer",
    }

    df["economic_view"] = pd.Categorical(df["economic_view"], categories=order, ordered=True)
    df = df.sort_values("economic_view")
    df["label"] = df["economic_view"].map(labels)

    x = range(len(df))

    plt.figure(figsize=(12, 7))
    plt.errorbar(
        x,
        df["p50"],
        yerr=[
            df["p50"] - df["p10"],
            df["p90"] - df["p50"],
        ],
        fmt="o",
        capsize=6,
        linewidth=2,
        label="Median with p10-p90 range",
    )

    plt.xticks(x, df["label"], rotation=20, ha="right")

    apply_article_style(
        title="12-Month Forecast Sensitivity by Economic View",
        xlabel="Economic View",
        ylabel="52-Week Mortgage Rate (%)",
    )

    for i, row in enumerate(df.itertuples()):
        plt.text(i, row.p50 + 0.06, f"{row.p50:.2f}%", ha="center", fontsize=10)

    plt.legend(frameon=False, fontsize=11)
    save_chart("06_economic_view_sensitivity.png")


def chart_economic_view_grid(mc_summary: pd.DataFrame) -> None:
    view_order = [
        "current_view",
        "higher_for_longer_view",
        "soft_landing_view",
        "recession_view",
        "stress_view",
    ]

    labels = {
        "current_view": "Current View",
        "higher_for_longer_view": "Higher-for-Longer",
        "soft_landing_view": "Soft Landing",
        "recession_view": "Recession",
        "stress_view": "Stress",
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, view in zip(axes, view_order):
        df = mc_summary[mc_summary["economic_view"] == view].copy()
        final = df[df["week"] == df["week"].max()].iloc[0]

        ax.plot(df["week"], df["p50"], linewidth=2.0, label="Median")
        ax.fill_between(df["week"], df["p25"], df["p75"], alpha=0.30, label="p25-p75")
        ax.fill_between(df["week"], df["p10"], df["p90"], alpha=0.18, label="p10-p90")
        ax.fill_between(df["week"], df["p05"], df["p95"], alpha=0.10, label="p05-p95")

        ax.set_title(
            f"{labels[view]}\n"
            f"52w median: {final['p50']:.2f}% | p10-p90: {final['p10']:.2f}%–{final['p90']:.2f}%",
            fontsize=12,
        )
        ax.set_xlabel("Weeks Ahead")
        ax.set_ylabel("Mortgage Rate (%)")
        ax.grid(True, alpha=0.25)

    axes[-1].axis("off")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False)

    fig.suptitle(
        "12-Month Mortgage Rate Forecast by Economic View",
        fontsize=18,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(MEDIUM_DIR / "07_economic_view_fan_chart_grid.png", dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {MEDIUM_DIR / '07_economic_view_fan_chart_grid.png'}")


def main() -> None:
    spine = pd.read_csv(SPINE_PATH, parse_dates=["week_date"])
    walk_forward = pd.read_csv(WALK_FORWARD_PATH, parse_dates=["week_date"])
    model_summary = pd.read_csv(DIRECT_VS_DECOMP_PATH)
    mc_summary = pd.read_csv(MONTE_CARLO_CURRENT_PATH)
    final_view = pd.read_csv(MONTE_CARLO_FINAL_VIEW_PATH)

    chart_decomposition(spine)
    chart_spread_mean_reversion(spine)
    chart_walk_forward(walk_forward)
    chart_model_comparison(model_summary)
    chart_monte_carlo_current_view(mc_summary)
    chart_economic_view_sensitivity(final_view)
    chart_economic_view_grid(mc_summary)

    print("\nMedium-ready charts saved to:")
    print(MEDIUM_DIR)


if __name__ == "__main__":
    main()
