from pathlib import Path
import pandas as pd
import numpy as np

WEEKLY_INPUT_PATH = Path("outputs/weekly_campaign_dataset.csv")
MODEL_INPUT_PATH = Path("outputs/campaign_model_dataset.csv")
SCENARIO_OUTPUT_PATH = Path("outputs/campaign_scenario_results.csv")


def load_weekly_campaign_data(path: Path = WEEKLY_INPUT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["week_start"])


def summarize_campaign_performance(weekly_campaign: pd.DataFrame) -> pd.DataFrame:
    campaign_perf = (
        weekly_campaign.groupby("campaign", as_index=False)
          .agg(
              spend=("spend", "sum"),
              revenue=("revenue", "sum"),
              conversions=("conversions", "sum"),
              clicks=("clicks", "sum"),
          )
    )

    campaign_perf["roas"] = np.where(
        campaign_perf["spend"] > 0,
        campaign_perf["revenue"] / campaign_perf["spend"],
        np.nan,
    )

    campaign_perf["conversion_rate"] = np.where(
        campaign_perf["clicks"] > 0,
        campaign_perf["conversions"] / campaign_perf["clicks"],
        np.nan,
    )

    return campaign_perf.sort_values("roas", ascending=False)


def campaign_to_col(name: str) -> str:
    return f"spend_{name.strip().lower().replace(' ', '_').replace('-', '_')}"


def simulate_campaign_budget_shift_model(
    model,
    model_df: pd.DataFrame,
    campaign_perf: pd.DataFrame,
    shift_pct: float = 0.10,
    top_n: int = 2,
    bottom_n: int = 2,
) -> dict:
    """Reallocates budget from bottom-ROAS to top-ROAS campaigns, then re-predicts
    revenue with the fitted Ridge model instead of assuming flat ROAS.
    """
    ranked = campaign_perf.sort_values("roas", ascending=False)
    top_campaigns = ranked.head(top_n)["campaign"].tolist()
    bottom_campaigns = ranked.tail(bottom_n)["campaign"].tolist()

    top_cols = [campaign_to_col(c) for c in top_campaigns]
    bottom_cols = [campaign_to_col(c) for c in bottom_campaigns]

    feature_cols = [c for c in model_df.columns if c not in ["week_start", "revenue"]]
    baseline_X = model_df[feature_cols].copy()
    scenario_X = baseline_X.copy()

    shift_amounts = baseline_X[bottom_cols] * shift_pct
    scenario_X[bottom_cols] = baseline_X[bottom_cols] - shift_amounts
    weekly_total_shift = shift_amounts.sum(axis=1)
    for col in top_cols:
        scenario_X[col] = baseline_X[col] + weekly_total_shift / len(top_cols)

    baseline_pred = model.predict(baseline_X)
    scenario_pred = model.predict(scenario_X)

    weekly_results = pd.DataFrame({
        "week_start": model_df["week_start"],
        "baseline_predicted_revenue": baseline_pred,
        "scenario_predicted_revenue": scenario_pred,
    })
    weekly_results["revenue_lift"] = (
        weekly_results["scenario_predicted_revenue"] - weekly_results["baseline_predicted_revenue"]
    )

    spend_summary = pd.DataFrame({
        "campaign": [c.replace("spend_", "") for c in feature_cols if c.startswith("spend_")],
        "current_spend": [baseline_X[c].sum() for c in feature_cols if c.startswith("spend_")],
        "scenario_spend": [scenario_X[c].sum() for c in feature_cols if c.startswith("spend_")],
    })

    totals = {
        "total_baseline_predicted_revenue": weekly_results["baseline_predicted_revenue"].sum(),
        "total_scenario_predicted_revenue": weekly_results["scenario_predicted_revenue"].sum(),
        "total_predicted_revenue_lift": weekly_results["revenue_lift"].sum(),
    }

    return {
        "weekly_results": weekly_results,
        "spend_summary": spend_summary,
        "totals": totals,
    }


def save_campaign_scenario_results(spend_summary: pd.DataFrame, path: Path = SCENARIO_OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    spend_summary.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    from src.model import fit_mmm_lite_campaign

    weekly_campaign = load_weekly_campaign_data(WEEKLY_INPUT_PATH)
    model_df = pd.read_csv(MODEL_INPUT_PATH, parse_dates=["week_start"])
    campaign_perf = summarize_campaign_performance(weekly_campaign)

    model, preds, metrics, coef_df = fit_mmm_lite_campaign(model_df)

    scenario = simulate_campaign_budget_shift_model(
        model=model,
        model_df=model_df,
        campaign_perf=campaign_perf,
        shift_pct=0.10,
        top_n=2,
        bottom_n=2,
    )

    save_path = save_campaign_scenario_results(scenario["spend_summary"], SCENARIO_OUTPUT_PATH)

    print(f"Saved scenario results to: {save_path}")
    print(scenario["totals"])
    print(scenario["spend_summary"])