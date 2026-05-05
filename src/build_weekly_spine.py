import pandas as pd
from project_paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


RAW_PATH = RAW_DATA_DIR / "fred_macro_raw.csv"
OUTPUT_DIR = PROCESSED_DATA_DIR

OUTPUT_PATH = OUTPUT_DIR / "weekly_macro_spine.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "weekly_macro_spine_diagnostics.csv"


DAILY_SERIES = [
    "DGS10",
    "DGS2",
    "DGS3MO",
    "DGS5",
    "DGS30",
    "DFII10",
    "T10YIE",
    "BAMLH0A0HYM2",
    "BAMLC0A4CBBB",
    "VIXCLS",
    "DTWEXBGS",
    "DCOILWTICO",
]

WEEKLY_SERIES = [
    "MORTGAGE30US",
    "NFCI",
]

MONTHLY_SERIES = [
    "FEDFUNDS",
    "BAA",
    "AAA",
    "CPIAUCSL",
    "CPILFESL",
    "PCEPI",
    "PCEPILFE",
    "UNRATE",
    "PAYEMS",
    "INDPRO",
    "HOUST",
    "PERMIT",
]


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw FRED file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, parse_dates=["observation_date"])

    required_cols = {"series_id", "observation_date", "value"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Raw file is missing required columns: {missing}")

    return df[["series_id", "observation_date", "value"]].copy()


def pivot_series(df: pd.DataFrame) -> pd.DataFrame:
    wide = (
        df.pivot_table(
            index="observation_date",
            columns="series_id",
            values="value",
            aggfunc="last",
        )
        .sort_index()
    )

    wide.index = pd.to_datetime(wide.index)

    return wide


def build_weekly_index(wide: pd.DataFrame) -> pd.DatetimeIndex:
    mortgage = wide["MORTGAGE30US"].dropna()

    if mortgage.empty:
        raise ValueError("MORTGAGE30US has no usable observations.")

    return pd.DatetimeIndex(mortgage.index).sort_values()


def align_daily_to_weekly(wide: pd.DataFrame, weekly_index: pd.DatetimeIndex) -> pd.DataFrame:
    daily = wide[[c for c in DAILY_SERIES if c in wide.columns]].copy()

    # Forward-fill daily values into mortgage-rate observation dates.
    # This uses the most recently available market value as of that weekly date.
    daily_aligned = daily.reindex(daily.index.union(weekly_index)).sort_index().ffill()
    daily_aligned = daily_aligned.loc[weekly_index]

    return daily_aligned


def align_weekly_to_weekly(wide: pd.DataFrame, weekly_index: pd.DatetimeIndex) -> pd.DataFrame:
    weekly = wide[[c for c in WEEKLY_SERIES if c in wide.columns]].copy()

    # Weekly series may end on different weekdays.
    # Forward-fill makes NFCI available as of the mortgage observation date.
    weekly_aligned = weekly.reindex(weekly.index.union(weekly_index)).sort_index().ffill()
    weekly_aligned = weekly_aligned.loc[weekly_index]

    return weekly_aligned


def align_monthly_to_weekly(wide: pd.DataFrame, weekly_index: pd.DatetimeIndex) -> pd.DataFrame:
    monthly = wide[[c for c in MONTHLY_SERIES if c in wide.columns]].copy()

    # Conservative no-leakage approximation:
    # monthly values are shifted one month before being made available.
    # Example: March CPI is not allowed to affect March weekly rows.
    monthly_lagged = monthly.copy()

    # Shift by ~4 weeks instead of 1 row
    monthly_lagged.index = monthly_lagged.index + pd.DateOffset(days=30)

    monthly_aligned = (
        monthly_lagged
        .reindex(monthly_lagged.index.union(weekly_index))
        .sort_index()
        .ffill()
    )

    monthly_aligned = monthly_aligned.loc[weekly_index]

    return monthly_aligned


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Core decomposition
    out["mortgage_spread"] = out["MORTGAGE30US"] - out["DGS10"]

    # Treasury curve
    out["yield_curve_10y_2y"] = out["DGS10"] - out["DGS2"]
    out["yield_curve_10y_3mo"] = out["DGS10"] - out["DGS3MO"]
    out["yield_curve_30y_10y"] = out["DGS30"] - out["DGS10"]

    # Inflation / real-rate structure
    out["breakeven_real_gap"] = out["T10YIE"] - out["DFII10"]

    # Credit spreads
    out["credit_spread_baa_aaa"] = out["BAA"] - out["AAA"]
    out["credit_spread_baa_10y"] = out["BAA"] - out["DGS10"]
    out["credit_spread_aaa_10y"] = out["AAA"] - out["DGS10"]

    # Weekly changes
    for col in [
        "MORTGAGE30US",
        "DGS10",
        "DGS2",
        "DGS3MO",
        "DGS5",
        "DGS30",
        "mortgage_spread",
        "VIXCLS",
        "NFCI",
    ]:
        if col in out.columns:
            out[f"{col.lower()}_chg_1w"] = out[col].diff(1)
            out[f"{col.lower()}_chg_4w"] = out[col].diff(4)

    # Rolling features
    for col in ["DGS10", "mortgage_spread", "VIXCLS", "NFCI"]:
        if col in out.columns:
            out[f"{col.lower()}_roll_mean_4w"] = out[col].rolling(4).mean()
            out[f"{col.lower()}_roll_mean_12w"] = out[col].rolling(12).mean()
            out[f"{col.lower()}_roll_std_4w"] = out[col].rolling(4).std()
            out[f"{col.lower()}_roll_std_12w"] = out[col].rolling(12).std()

    # Mean reversion feature for spread model
    out["mortgage_spread_roll_mean_5y"] = out["mortgage_spread"].rolling(52 * 5).mean()
    out["mortgage_spread_gap_vs_5y_avg"] = (
        out["mortgage_spread"] - out["mortgage_spread_roll_mean_5y"]
    )

    # Targets
    out["target_dgs10_chg_1w"] = out["DGS10"].shift(-1) - out["DGS10"]
    out["target_dgs10_chg_4w"] = out["DGS10"].shift(-4) - out["DGS10"]
    out["target_dgs10_chg_12w"] = out["DGS10"].shift(-12) - out["DGS10"]

    out["target_spread_chg_1w"] = out["mortgage_spread"].shift(-1) - out["mortgage_spread"]
    out["target_spread_chg_4w"] = out["mortgage_spread"].shift(-4) - out["mortgage_spread"]
    out["target_spread_chg_12w"] = out["mortgage_spread"].shift(-12) - out["mortgage_spread"]

    out["target_mortgage_rate_chg_1w"] = out["MORTGAGE30US"].shift(-1) - out["MORTGAGE30US"]
    out["target_mortgage_rate_chg_4w"] = out["MORTGAGE30US"].shift(-4) - out["MORTGAGE30US"]
    out["target_mortgage_rate_chg_12w"] = out["MORTGAGE30US"].shift(-12) - out["MORTGAGE30US"]

    return out


def build_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    diagnostics = []

    for col in df.columns:
        diagnostics.append(
            {
                "column_name": col,
                "rows": len(df),
                "non_null_values": df[col].notna().sum(),
                "null_values": df[col].isna().sum(),
                "min_value": df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
                "max_value": df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None,
            }
        )

    return pd.DataFrame(diagnostics)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    wide = pivot_series(raw)

    weekly_index = build_weekly_index(wide)

    daily = align_daily_to_weekly(wide, weekly_index)
    weekly = align_weekly_to_weekly(wide, weekly_index)
    monthly = align_monthly_to_weekly(wide, weekly_index)

    spine = pd.concat([weekly, daily, monthly], axis=1)
    spine.index.name = "week_date"

    spine = add_engineered_features(spine)
    spine = spine.reset_index()

    spine.to_csv(OUTPUT_PATH, index=False)

    diagnostics = build_diagnostics(spine)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)

    print(f"Saved weekly spine to: {OUTPUT_PATH}")
    print(f"Saved diagnostics to: {DIAGNOSTICS_PATH}")

    print("\nShape:")
    print(spine.shape)

    print("\nDate range:")
    print(spine["week_date"].min(), "to", spine["week_date"].max())

    print("\nPreview:")
    print(spine.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
