from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

INPUT_PATH = Path("data/processed/final_dataset.csv")
WEEKLY_OUTPUT_PATH = Path("outputs/weekly_campaign_dataset.csv")
MODEL_OUTPUT_PATH = Path("outputs/campaign_model_dataset.csv")
CAMPAIGN_CONTRIB_PATH = Path("outputs/campaign_contributions.csv")


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where((den.isna()) | (den == 0), np.nan, num / den)


def load_clean_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date", "week_start"])


def build_weekly_campaign_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "week_start" not in df.columns and "date" in df.columns:
        df["week_start"] = pd.to_datetime(df["date"]).dt.to_period("W").apply(lambda r: r.start_time)

    weekly = (
        df.groupby(["week_start", "campaign"], as_index=False)
          .agg(
              spend=("cost", "sum"),
              revenue=("revenue", "sum"),
              conversions=("conversions", "sum"),
              clicks=("clicks", "sum"),
              impressions=("impressions", "sum"),
          )
    )

    weekly["roas"] = safe_div(weekly["revenue"], weekly["spend"])
    weekly["conversion_rate"] = safe_div(weekly["conversions"], weekly["clicks"])
    weekly["ctr"] = safe_div(weekly["clicks"], weekly["impressions"])

    return weekly


def build_campaign_model_dataset(weekly_campaign: pd.DataFrame) -> pd.DataFrame:
    model_df = weekly_campaign.pivot_table(
        index="week_start",
        columns="campaign",
        values="spend",
        aggfunc="sum",
        fill_value=0,
    )

    model_df.columns = [
        f"spend_{str(col).strip().lower().replace(' ', '_').replace('-', '_')}"
        for col in model_df.columns
    ]

    revenue_by_week = weekly_campaign.groupby("week_start", as_index=True)["revenue"].sum()

    model_df = model_df.join(revenue_by_week)
    model_df = model_df.sort_index().reset_index()
    model_df["time_index"] = np.arange(len(model_df))

    return model_df


def fit_mmm_lite_campaign(model_df: pd.DataFrame):
    X_cols = [c for c in model_df.columns if c not in ["week_start", "revenue"]]
    X = model_df[X_cols]
    y = model_df["revenue"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X)

    metrics = {
        "r2": r2_score(y, preds),
        "mae": mean_absolute_error(y, preds),
    }

    coef_df = pd.DataFrame({
        "feature": X_cols,
        "coefficient": model.coef_,
    }).sort_values("coefficient", ascending=False)

    return model, preds, metrics, coef_df


def export_model_artifacts(
    weekly_campaign: pd.DataFrame,
    model_df: pd.DataFrame,
    coef_df: pd.DataFrame,
):
    WEEKLY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAMPAIGN_CONTRIB_PATH.parent.mkdir(parents=True, exist_ok=True)

    weekly_campaign.to_csv(WEEKLY_OUTPUT_PATH, index=False)
    model_df.to_csv(MODEL_OUTPUT_PATH, index=False)
    coef_df.to_csv(CAMPAIGN_CONTRIB_PATH, index=False)


if __name__ == "__main__":
    df = load_clean_data(INPUT_PATH)
    weekly_campaign = build_weekly_campaign_dataset(df)
    model_df = build_campaign_model_dataset(weekly_campaign)
    model, preds, metrics, coef_df = fit_mmm_lite_campaign(model_df)
    export_model_artifacts(weekly_campaign, model_df, coef_df)

    print(metrics)
    print(coef_df.head())

def evaluate_model_cv(model_df: pd.DataFrame, alpha: float = 1.0) -> dict:
    """Leave-one-out CV — appropriate given only ~27 weekly rows.
    In-sample R²/MAE are optimistic; this gives an honest generalization estimate.
    """
    X_cols = [c for c in model_df.columns if c not in ["week_start", "revenue"]]
    X = model_df[X_cols]
    y = model_df["revenue"]

    cv_preds = cross_val_predict(Ridge(alpha=alpha), X, y, cv=LeaveOneOut())

    return {
        "cv_r2": r2_score(y, cv_preds),
        "cv_mae": mean_absolute_error(y, cv_preds),
    }
def fit_mmm_lite_campaign(model_df: pd.DataFrame, alpha: float = 1.0):
    X_cols = [c for c in model_df.columns if c not in ["week_start", "revenue"]]
    X = model_df[X_cols]
    y = model_df["revenue"]

    model = Ridge(alpha=alpha)
    model.fit(X, y)
    preds = model.predict(X)

    metrics = {
        "r2": r2_score(y, preds),
        "mae": mean_absolute_error(y, preds),
    }

    coef_df = pd.DataFrame({
        "feature": X_cols,
        "coefficient": model.coef_,
    }).sort_values("coefficient", ascending=False)

    return model, preds, metrics, coef_df