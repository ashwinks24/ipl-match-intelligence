import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def load_base_data() -> pd.DataFrame:
    df = pd.read_parquet(
        PROCESSED_DIR / "ball_by_ball.parquet"
    )

    df = df.sort_values(
        ["match_id", "innings", "ball"]
    )

    df["over"] = (
        df["ball"].astype(int)
    )

    return df


def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every match:
    target = first innings score + 1
    """

    first_innings_total = (
        df[df["innings"] == 1]
        .groupby("match_id")["total_runs_ball"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "total_runs_ball":
                "first_innings_total"
            }
        )
    )

    df = df.merge(
        first_innings_total,
        on="match_id",
        how="left"
    )

    df["target"] = (
        df["first_innings_total"] + 1
    )

    df["target"] = (
        df["target"].fillna(0)
    )

    return df


def add_innings_state_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match situation features.
    """

    # Current run rate
    df["current_run_rate"] = np.where(
        df["overs_completed"] > 0,
        df["cum_runs"] / df["overs_completed"],
        0
    )

    # Phase of innings
    df["over_phase"] = pd.cut(
        df["over"],
        bins=[-1, 5, 14, 19],
        labels=[
            "powerplay",
            "middle",
            "death"
        ]
    )

    # Wickets available
    df["wickets_in_hand"] = (
        10 - df["cum_wickets"]
    )

    # Runs still needed
    df["required_runs"] = np.where(
        df["innings"] == 2,
        (df["target"] - df["cum_runs"]).clip(lower=0),
        0
    )

    # Required run rate
    df["required_run_rate"] = np.where(
        (df["innings"] == 2)
        & (df["balls_remaining"] > 0),

        df["required_runs"]
        / (df["balls_remaining"] / 6),

        0
    )
    df["required_run_rate"]=df["required_run_rate"].clip(upper=36);

    # Pressure = required RR - current RR
    df["run_rate_pressure"] = np.where(
        df["innings"] == 2,

        df["required_run_rate"]
        - df["current_run_rate"],

        0
    )

    # Percentage innings completed
    df["balls_used_pct"] = (
        df["cum_legal_balls"] / 120
    )

    # Percentage wickets lost
    df["wickets_lost_pct"] = (
        df["cum_wickets"] / 10
    )

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recent scoring patterns and momentum indicators.
    """

    g = ["match_id", "innings"]

    # Runs scored in last 6 balls
    df["runs_last_6"] = (
        df.groupby(g)["total_runs_ball"]
        .transform(
            lambda x: x.rolling(
                6,
                min_periods=1
            ).sum()
        )
    )

    # Runs scored in last 18 balls
    df["runs_last_18"] = (
        df.groupby(g)["total_runs_ball"]
        .transform(
            lambda x: x.rolling(
                18,
                min_periods=1
            ).sum()
        )
    )

    # Wickets in last 18 balls
    df["wickets_last_18"] = (
        df.groupby(g)["is_wicket"]
        .transform(
            lambda x: x.rolling(
                18,
                min_periods=1
            ).sum()
        )
    )

    # Dot balls
    df["is_dot"] = (
        df["total_runs_ball"] == 0
    ).astype(int)

    df["dots_last_6"] = (
        df.groupby(g)["is_dot"]
        .transform(
            lambda x: x.rolling(
                6,
                min_periods=1
            ).sum()
        )
    )

    # Boundary indicator
    df["is_boundary"] = (
        df["is_four"] + df["is_six"]
    ).clip(upper=1)

    # Boundaries in last 6 balls
    df["boundaries_last_6"] = (
        df.groupby(g)["is_boundary"]
        .transform(
            lambda x: x.rolling(
                6,
                min_periods=1
            ).sum()
        )
    )

    # Partnership runs
    def partnership_runs(group):

        result = []
        current = 0

        for _, row in group.iterrows():

            current += row["total_runs_ball"]

            result.append(current)

            if row["is_wicket"] == 1:
                current = 0

        return pd.Series(
            result,
            index=group.index
        )

    df["partnership_runs"] = (
        df.groupby(g)
        .apply(partnership_runs)
        .reset_index(
            level=[0, 1],
            drop=True
        )
    )

    # Partnership balls
    def partnership_balls(group):

        result = []
        current = 0

        for _, row in group.iterrows():

            current += row["is_legal_ball"]

            result.append(current)

            if row["is_wicket"] == 1:
                current = 0

        return pd.Series(
            result,
            index=group.index
        )

    df["partnership_balls"] = (
        df.groupby(g)
        .apply(partnership_balls)
        .reset_index(
            level=[0, 1],
            drop=True
        )
    )

    # Momentum score
    df["momentum_index"] = (
        df["runs_last_6"]
        - (2 * df["wickets_last_18"])
    )

    # Boundary pressure
    df["boundary_pressure"] = (
        df["boundaries_last_6"] / 6
    )

    return df




def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical venue scoring trends.
    Uses only seasons before the current season.
    """

    if "season" not in df.columns:

        if "season_x" in df.columns:
            df["season"] = df["season_x"]

        elif "season_y" in df.columns:
            df["season"] = df["season_y"]

    if "venue" not in df.columns:

        if "venue_x" in df.columns:
            df["venue"] = df["venue_x"]

        elif "venue_y" in df.columns:
            df["venue"] = df["venue_y"]

    df["season"] = (
    df["season"]
    .astype(str)
    .str[:4]
    )

    df["season"] = pd.to_numeric(
    df["season"],
    errors="coerce"
    )

    venue_scores = (
        df[df["innings"] == 1]
        .groupby(
            ["venue", "match_id", "season"]
        )["total_runs_ball"]
        .sum()
        .reset_index()
    )

    venue_hist = {}

    seasons = sorted(
        venue_scores["season"]
        .dropna()
        .unique()
    )

    for season in seasons:

        past_data = venue_scores[
            venue_scores["season"] < season
        ]

        venue_hist[season] = (
            past_data
            .groupby("venue")["total_runs_ball"]
            .mean()
        )

    df["venue_avg_score"] = np.nan

    for season in seasons:

        mask = df["season"] == season

        df.loc[mask, "venue_avg_score"] = (
            df.loc[mask, "venue"]
            .map(venue_hist[season])
        )

    overall_venue_avg = (
        venue_scores
        .groupby("venue")["total_runs_ball"]
        .mean()
    )

    df["venue_avg_score"] = (
        df["venue_avg_score"]
        .fillna(
            df["venue"].map(
                overall_venue_avg
            )
        )
    )

    return df


def add_match_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every ball, did the batting team
    eventually win the match?
    """

    df = df[df["winner"].notna()].copy()

    df["batting_team_won"] = (
        df["batting_team"]
        == df["winner"]
    ).astype(int)

    return df

def build_features(save: bool = True) -> pd.DataFrame:
    print("Loading base data...")
    df = load_base_data()

    if "season_x" in df.columns:
        df["season"] = df["season_x"]

    if "venue_x" in df.columns:
        df["venue"] = df["venue_x"]

    df = df.drop(
    columns=[
        c
        for c in [
            "season_x",
            "season_y",
            "venue_x",
            "venue_y"
        ]
        if c in df.columns
    ]
)

    print("Adding target column...")
    df = add_target_column(df)

    print("Adding innings state features...")
    df = add_innings_state_features(df)

    print("Adding momentum features...")
    df = add_momentum_features(df)

    print("Adding historical features...")
    df = add_historical_features(df)

    print("Adding match labels...")
    df = add_match_label(df)

    drop_cols = [
    
    
    "other_wicket_type",
    "other_player_dismissed",
    "penalty",
    "winner_runs",
    "winner_wickets",
    "outcome",
    "eliminator",
    "method"
    ]

    df = df.drop(
        columns=[
            c for c in drop_cols
            if c in df.columns
        ]
    )

    print("\nFINAL FEATURE SUMMARY")
    print(f"Shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print(
        f"Target distribution:\n"
        f"{df['batting_team_won'].value_counts()}"
    )
    print(
        f"Null counts:\n"
        f"{df.isnull().sum()[df.isnull().sum() > 0]}"
    )
    print("\nRemaining season nulls:")
    print(df["season"].isna().sum())

    print("\nRemaining venue_avg_score nulls:")
    print(df["venue_avg_score"].isna().sum())

    if save:
        out = PROCESSED_DIR / "features.parquet"

        df.to_parquet(
            out,
            index=False
        )

        print(f"\nSaved → {out}")

    return df


if __name__ == "__main__":
    df = build_features()