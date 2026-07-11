from pathlib import Path
import pandas as pd
import numpy as np

RAW_PATH = Path("/Users/erincheng/Desktop/media-analytics-mmm-lite/data/raw/media_facebook.csv")
PROCESSED_PATH = Path("/Users/erincheng/Desktop/media-analytics-mmm-lite/data/processed/final_dataset.csv")

NUMERIC_COLS = [
    "impressions", "clicks", "conversions",
    "cost", "revenue",
    "cpc", "cpa", "ctr",
    "conversion_rate", "roas", "roi", "profit_margin"
]

BUSINESS_KEY = ["date", "channel", "campaign"]


def to_snake(col: str) -> str:
    col = col.strip().lower()
    for ch in [" ", "-", "/", "(", ")", "%"]:
        col = col.replace(ch, "_")
    while "__" in col:
        col = col.replace("__", "_")
    return col.strip("_")


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where((den.isna()) | (den == 0), np.nan, num / den)


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [to_snake(c) for c in df.columns]
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def remove_duplicates_and_invalids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    existing_keys = [c for c in BUSINESS_KEY if c in df.columns]

    if existing_keys:
        df = df.drop_duplicates(subset=existing_keys)
    else:
        df = df.drop_duplicates()

    for col in ["impressions", "clicks", "conversions", "cost", "revenue"]:
        if col in df.columns:
            df = df[df[col].fillna(0) >= 0]

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_dims = [c for c in ["date", "channel", "campaign"] if c in df.columns]
    if required_dims:
        df = df.dropna(subset=required_dims)

    for col in ["impressions", "clicks", "conversions", "cost", "revenue"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"cost", "clicks"}.issubset(df.columns):
        df["cpc_calc"] = safe_div(df["cost"], df["clicks"])

    if {"cost", "conversions"}.issubset(df.columns):
        df["cpa_calc"] = safe_div(df["cost"], df["conversions"])

    if {"clicks", "impressions"}.issubset(df.columns):
        df["ctr_calc"] = safe_div(df["clicks"], df["impressions"])

    if {"conversions", "clicks"}.issubset(df.columns):
        df["conversion_rate_calc"] = safe_div(df["conversions"], df["clicks"])

    if {"revenue", "cost"}.issubset(df.columns):
        df["roas_calc"] = safe_div(df["revenue"], df["cost"])
        df["roi_calc"] = safe_div(df["revenue"] - df["cost"], df["cost"])
        df["profit_margin_calc"] = safe_div(df["revenue"] - df["cost"], df["revenue"])
        df["profit"] = df["revenue"] - df["cost"]

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" in df.columns:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.month_name()
        df["quarter"] = df["date"].dt.to_period("Q").astype(str)
        df["week_start"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    return df


def validate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    metric_pairs = [
        ("cpc", "cpc_calc"),
        ("cpa", "cpa_calc"),
        ("ctr", "ctr_calc"),
        ("conversion_rate", "conversion_rate_calc"),
        ("roas", "roas_calc"),
        ("roi", "roi_calc"),
        ("profit_margin", "profit_margin_calc"),
    ]

    for source_col, calc_col in metric_pairs:
        if source_col in df.columns and calc_col in df.columns:
            checks.append({
                "metric": source_col,
                "avg_abs_diff": (df[source_col] - df[calc_col]).abs().mean()
            })

    return pd.DataFrame(checks)


def clean_media_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = load_raw_data(path)
    df = cast_types(df)
    df = remove_duplicates_and_invalids(df)
    df = handle_missing_values(df)
    df = add_kpis(df)
    df = add_time_features(df)
    return df


def save_clean_data(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    df = clean_media_data(RAW_PATH)
    save_path = save_clean_data(df, PROCESSED_PATH)
    print(f"Saved cleaned dataset to: {save_path}")
    print(validate_kpis(df))