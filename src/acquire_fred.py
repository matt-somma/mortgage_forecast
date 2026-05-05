import os
from pathlib import Path
from datetime import datetime, timezone
import time

import pandas as pd
import requests
from project_paths import PROJECT_ROOT, RAW_DATA_DIR


DEFAULT_KEY_PATH = PROJECT_ROOT / ".secrets" / "fred_api_key.txt"
OUTPUT_DIR = RAW_DATA_DIR

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"

SERIES = {
    # Targets / decomposition
    "MORTGAGE30US": "30-Year Fixed Mortgage Rate",
    "DGS10": "10-Year Treasury Yield",

    # Treasury curve / rate environment
    "DGS2": "2-Year Treasury Yield",
    "DGS3MO": "3-Month Treasury Yield",
    "DGS5": "5-Year Treasury Yield",
    "DGS30": "30-Year Treasury Yield",
    "DFII10": "10-Year TIPS Real Yield",
    "T10YIE": "10-Year Breakeven Inflation Rate",
    "FEDFUNDS": "Federal Funds Rate",

    # Credit / spread / risk proxies
    "BAA": "Moody's BAA Corporate Bond Yield",
    "AAA": "Moody's AAA Corporate Bond Yield",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "BAMLC0A4CBBB": "ICE BofA BBB US Corporate OAS",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "VIXCLS": "CBOE Volatility Index",

    # Inflation / macro
    "CPIAUCSL": "CPI",
    "CPILFESL": "Core CPI",
    "PCEPI": "PCE Price Index",
    "PCEPILFE": "Core PCE Price Index",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",

    # Growth / housing
    "INDPRO": "Industrial Production Index",
    "HOUST": "Housing Starts",
    "PERMIT": "Building Permits",

    # External pressure
    "DTWEXBGS": "Nominal Broad U.S. Dollar Index",
    "DCOILWTICO": "WTI Crude Oil Price",
}


def read_api_key() -> str:
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if api_key:
        return api_key

    key_path = Path(os.environ.get("FRED_API_KEY_PATH", DEFAULT_KEY_PATH)).expanduser()
    if not key_path.exists():
        raise FileNotFoundError(f"FRED API key file not found: {key_path}")

    api_key = key_path.read_text(encoding="utf-8").strip()

    if not api_key:
        raise ValueError(f"FRED API key file is empty: {key_path}")

    return api_key


def fred_get(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"FRED request failed: {response.status_code} {response.text}"
        ) from exc

    payload = response.json()

    if "error_code" in payload:
        raise RuntimeError(
            f"FRED API error {payload.get('error_code')}: {payload.get('error_message')}"
        )

    return payload


def fetch_series_metadata(series_id: str, api_key: str) -> dict:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }

    payload = fred_get(FRED_SERIES_URL, params)
    series = payload.get("seriess", [])

    if not series:
        return {
            "series_id": series_id,
            "title": SERIES.get(series_id),
            "frequency": None,
            "units": None,
            "seasonal_adjustment": None,
            "last_updated": None,
        }

    item = series[0]

    return {
        "series_id": series_id,
        "title": item.get("title"),
        "frequency": item.get("frequency"),
        "units": item.get("units"),
        "seasonal_adjustment": item.get("seasonal_adjustment"),
        "last_updated": item.get("last_updated"),
    }


def fetch_series_observations(
    series_id: str,
    api_key: str,
    start_date: str = "1990-01-01",
) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    payload = fred_get(FRED_OBSERVATIONS_URL, params)
    observations = payload.get("observations", [])

    if not observations:
        raise ValueError(f"No observations returned for {series_id}")

    df = pd.DataFrame(observations)

    df = df.rename(
        columns={
            "date": "observation_date",
            "value": "value_raw",
        }
    )

    df["series_id"] = series_id
    df["series_label"] = SERIES.get(series_id)
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value_raw"], errors="coerce")
    df["realtime_start"] = pd.to_datetime(df["realtime_start"], errors="coerce")
    df["realtime_end"] = pd.to_datetime(df["realtime_end"], errors="coerce")
    df["ingested_at_utc"] = datetime.now(timezone.utc)

    return df[
        [
            "series_id",
            "series_label",
            "observation_date",
            "value",
            "value_raw",
            "realtime_start",
            "realtime_end",
            "ingested_at_utc",
        ]
    ]


def build_diagnostics(df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    diagnostics = (
        df.groupby("series_id")
        .agg(
            rows=("value_raw", "size"),
            non_null_values=("value", "count"),
            null_values=("value", lambda s: s.isna().sum()),
            min_date=("observation_date", "min"),
            max_date=("observation_date", "max"),
            min_value=("value", "min"),
            max_value=("value", "max"),
        )
        .reset_index()
    )

    diagnostics = diagnostics.merge(metadata_df, on="series_id", how="left")

    return diagnostics[
        [
            "series_id",
            "title",
            "frequency",
            "units",
            "seasonal_adjustment",
            "rows",
            "non_null_values",
            "null_values",
            "min_date",
            "max_date",
            "min_value",
            "max_value",
            "last_updated",
        ]
    ]


def fetch_all(start_date: str = "1990-01-01", sleep_seconds: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    api_key = read_api_key()

    observation_frames = []
    metadata_rows = []
    failures = []

    for series_id in SERIES:
        print(f"Fetching {series_id}...")

        try:
            metadata_rows.append(fetch_series_metadata(series_id, api_key))
            observation_frames.append(fetch_series_observations(series_id, api_key, start_date))
        except Exception as exc:
            failures.append({"series_id": series_id, "error": str(exc)})
            print(f"  FAILED: {series_id} | {exc}")

        time.sleep(sleep_seconds)

    if not observation_frames:
        raise RuntimeError("No series were successfully fetched.")

    observations_df = pd.concat(observation_frames, ignore_index=True)
    metadata_df = pd.DataFrame(metadata_rows)

    if failures:
        failure_df = pd.DataFrame(failures)
        failure_path = OUTPUT_DIR / "fred_acquisition_failures.csv"
        failure_df.to_csv(failure_path, index=False)
        print(f"\nSome series failed. Failure details saved to: {failure_path}")

    return observations_df, metadata_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    observations_df, metadata_df = fetch_all(start_date="1990-01-01")

    raw_path = OUTPUT_DIR / "fred_macro_raw.csv"
    metadata_path = OUTPUT_DIR / "fred_series_metadata.csv"
    diagnostics_path = OUTPUT_DIR / "fred_acquisition_diagnostics.csv"

    (observations_df.sort_values(["series_id", "observation_date"]).to_csv(raw_path, index=False))
    metadata_df.to_csv(metadata_path, index=False)

    diagnostics_df = build_diagnostics(observations_df, metadata_df)
    diagnostics_df.to_csv(diagnostics_path, index=False)

    print("\nSaved files:")
    print(f"  Raw observations: {raw_path}")
    print(f"  Series metadata:  {metadata_path}")
    print(f"  Diagnostics:      {diagnostics_path}")

    print("\nDiagnostics preview:")
    print(diagnostics_df.sort_values("series_id").to_string(index=False))


if __name__ == "__main__":
    main()
