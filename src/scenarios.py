from pathlib import Path
import pandas as pd
import numpy as np

WEEKLY_INPUT_PATH = Path("outputs/weekly_campaign_dataset.csv")
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


def simulate_campaign_budget_shift(
    campaign_perf: pd.DataFrame,
    shift_pct: float = 0.10,
    top_n: int = 2,
    bottom_n: int = 2,
) -> pd.DataFrame:
    campaign_perf = campaign_perf.sort_values("roas", ascending=False).copy()

    if campaign_perf["campaign"].nunique() < 2:
        raise ValueError("Need at least 2 campaigns for campaign-level budget reallocation.")

    top_campaigns = campaign_perf.head(top_n)["campaign"].tolist()
    bottom_campaigns = campaign_perf.tail(bottom_n)["campaign"].tolist()

    current_budget = campaign_perf.set_index("campaign")["spend"].to_dict()
    scenario_budget = current_budget.copy()

    total_shift = 0
    for campaign in bottom_campaigns:
        shift_amt = current_budget[campaign] * shift_pct
        scenario_budget[campaign] -= shift_amt
        total_shift += shift_amt

    for campaign in top_campaigns:
        scenario_budget[campaign] += total_shift / len(top_campaigns)

    scenario_df = pd.DataFrame({
        "campaign": list(current_budget.keys()),
        "current_spend": list(current_budget.values()),
        "scenario_spend": [scenario_budget[c] for c in current_budget.keys()],
    })

    scenario_df = scenario_df.merge(
        campaign_perf[["campaign", "roas", "conversion_rate"]],
        on="campaign",
        how="left",
    )

    scenario_df["projected_revenue_current"] = scenario_df["current_spend"] * scenario_df["roas"]
    scenario_df["projected_revenue_scenario"] = scenario_df["scenario_spend"] * scenario_df["roas"]
    scenario_df["projected_revenue_lift"] = (
        scenario_df["projected_revenue_scenario"] - scenario_df["projected_revenue_current"]
    )

    return scenario_df.sort_values("projected_revenue_lift", ascending=False)


def save_campaign_scenario_results(df: pd.DataFrame, path: Path = SCENARIO_OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    weekly_campaign = load_weekly_campaign_data(WEEKLY_INPUT_PATH)
    campaign_perf = summarize_campaign_performance(weekly_campaign)
    scenario_df = simulate_campaign_budget_shift(
        campaign_perf,
        shift_pct=0.10,
        top_n=2,
        bottom_n=2,
    )
    save_path = save_campaign_scenario_results(scenario_df, SCENARIO_OUTPUT_PATH)

    print(f"Saved scenario results to: {save_path}")
    print(scenario_df)