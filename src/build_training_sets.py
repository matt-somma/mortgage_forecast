import pandas as pd
from project_paths import MODELING_DATA_DIR, PROCESSED_DATA_DIR


SPINE_PATH = PROCESSED_DATA_DIR / "weekly_macro_spine.csv"
OUTPUT_DIR = MODELING_DATA_DIR


TARGETS = {
    "dgs10_1w": "target_dgs10_chg_1w",
    "dgs10_4w": "target_dgs10_chg_4w",
    "dgs10_12w": "target_dgs10_chg_12w",
    "spread_1w": "target_spread_chg_1w",
    "spread_4w": "target_spread_chg_4w",
    "spread_12w": "target_spread_chg_12w",
}


DGS10_FEATURES_LONG = [
    "DGS10",
    "DGS2",
    "DGS3MO",
    "DGS5",
    "DGS30",
    "DFII10",
    "T10YIE",
    "FEDFUNDS",
    "CPIAUCSL",
    "CPILFESL",
    "PCEPI",
    "PCEPILFE",
    "UNRATE",
    "PAYEMS",
    "INDPRO",
    "VIXCLS",
    "yield_curve_10y_2y",
    "yield_curve_10y_3mo",
    "yield_curve_30y_10y",
    "breakeven_real_gap",
    "dgs10_chg_1w",
    "dgs10_chg_4w",
    "dgs2_chg_1w",
    "dgs2_chg_4w",
    "dgs3mo_chg_1w",
    "dgs3mo_chg_4w",
    "dgs5_chg_1w",
    "dgs5_chg_4w",
    "dgs30_chg_1w",
    "dgs30_chg_4w",
    "dgs10_roll_mean_4w",
    "dgs10_roll_mean_12w",
    "dgs10_roll_std_4w",
    "dgs10_roll_std_12w",
    "vixcls_chg_1w",
    "vixcls_chg_4w",
    "vixcls_roll_mean_4w",
    "vixcls_roll_mean_12w",
    "vixcls_roll_std_4w",
    "vixcls_roll_std_12w",
]


SPREAD_FEATURES_LONG = [
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
    "dgs10_roll_mean_4w",
    "dgs10_roll_mean_12w",
    "dgs10_roll_std_4w",
    "dgs10_roll_std_12w",
    "vixcls_chg_1w",
    "vixcls_chg_4w",
    "vixcls_roll_mean_4w",
    "vixcls_roll_mean_12w",
    "vixcls_roll_std_4w",
    "vixcls_roll_std_12w",
    "nfci_chg_1w",
    "nfci_chg_4w",
    "nfci_roll_mean_4w",
    "nfci_roll_mean_12w",
    "nfci_roll_std_4w",
    "nfci_roll_std_12w",
]


SHORT_HISTORY_EXTRA_FEATURES = [
    "BAMLH0A0HYM2",
    "BAMLC0A4CBBB",
]


def load_spine() -> pd.DataFrame:
    if not SPINE_PATH.exists():
        raise FileNotFoundError(f"Weekly spine not found: {SPINE_PATH}")

    df = pd.read_csv(SPINE_PATH, parse_dates=["week_date"])
    df = df.sort_values("week_date").reset_index(drop=True)

    return df


def keep_existing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def add_time_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["split"] = "train"
    out.loc[out["week_date"] >= "2019-01-01", "split"] = "validation"
    out.loc[out["week_date"] >= "2023-01-01", "split"] = "test"

    return out


def build_training_set(
    df: pd.DataFrame,
    dataset_name: str,
    target_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    cols = ["week_date"] + feature_cols + [target_col]

    model_df = df[cols].copy()
    model_df = model_df.dropna(subset=[target_col])

    # Drop rows where core selected features are missing.
    model_df = model_df.dropna(subset=feature_cols)

    model_df = add_time_split(model_df)

    model_df = model_df.rename(columns={target_col: "target"})

    model_df.insert(1, "dataset_name", dataset_name)

    return model_df


def build_feature_manifest(dataset_name: str, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_name": dataset_name,
            "target_col": target_col,
            "feature_name": feature_cols,
        }
    )


def build_diagnostics(dataset_name: str, model_df: pd.DataFrame) -> dict:
    return {
        "dataset_name": dataset_name,
        "rows": len(model_df),
        "min_week": model_df["week_date"].min(),
        "max_week": model_df["week_date"].max(),
        "train_rows": (model_df["split"] == "train").sum(),
        "validation_rows": (model_df["split"] == "validation").sum(),
        "test_rows": (model_df["split"] == "test").sum(),
        "target_mean": model_df["target"].mean(),
        "target_std": model_df["target"].std(),
        "target_min": model_df["target"].min(),
        "target_max": model_df["target"].max(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_spine()

    dgs10_long_features = keep_existing_columns(df, DGS10_FEATURES_LONG)
    spread_long_features = keep_existing_columns(df, SPREAD_FEATURES_LONG)

    dgs10_short_features = keep_existing_columns(
        df, DGS10_FEATURES_LONG + SHORT_HISTORY_EXTRA_FEATURES
    )
    spread_short_features = keep_existing_columns(
        df, SPREAD_FEATURES_LONG + SHORT_HISTORY_EXTRA_FEATURES
    )

    jobs = [
        ("dgs10_long_1w", TARGETS["dgs10_1w"], dgs10_long_features),
        ("dgs10_long_4w", TARGETS["dgs10_4w"], dgs10_long_features),
        ("dgs10_long_12w", TARGETS["dgs10_12w"], dgs10_long_features),
        ("spread_long_1w", TARGETS["spread_1w"], spread_long_features),
        ("spread_long_4w", TARGETS["spread_4w"], spread_long_features),
        ("spread_long_12w", TARGETS["spread_12w"], spread_long_features),

        # Short-history versions include modern OAS features.
        ("dgs10_short_oas_1w", TARGETS["dgs10_1w"], dgs10_short_features),
        ("spread_short_oas_1w", TARGETS["spread_1w"], spread_short_features),
    ]

    diagnostics = []
    manifests = []

    for dataset_name, target_col, feature_cols in jobs:
        model_df = build_training_set(
            df=df,
            dataset_name=dataset_name,
            target_col=target_col,
            feature_cols=feature_cols,
        )

        output_path = OUTPUT_DIR / f"{dataset_name}.csv"
        model_df.to_csv(output_path, index=False)

        diagnostics.append(build_diagnostics(dataset_name, model_df))
        manifests.append(build_feature_manifest(dataset_name, target_col, feature_cols))

        print(f"Saved {dataset_name}: {output_path} | rows={len(model_df)}")

    diagnostics_df = pd.DataFrame(diagnostics)
    manifest_df = pd.concat(manifests, ignore_index=True)

    diagnostics_path = OUTPUT_DIR / "training_set_diagnostics.csv"
    manifest_path = OUTPUT_DIR / "feature_manifest.csv"

    diagnostics_df.to_csv(diagnostics_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    print("\nSaved diagnostics:")
    print(f"  {diagnostics_path}")
    print(f"  {manifest_path}")

    print("\nDiagnostics preview:")
    print(diagnostics_df.to_string(index=False))


if __name__ == "__main__":
    main()
