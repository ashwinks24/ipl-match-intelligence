import pandas as pd
import numpy as np
import shap
import pickle
from pathlib import Path
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)

PHASE_MAP = {
    "powerplay": 0,
    "middle": 1,
    "death": 2
}


def load_artifacts():

    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)

    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    return model, calibrator, feature_cols


def load_data():

    df = pd.read_parquet(
        PROCESSED_DIR / "features.parquet"
    )

    df = df[
        df["innings"] == 2
    ].copy()

    if "over_phase" in df.columns:

        df["over_phase"] = (
            df["over_phase"]
            .map(PHASE_MAP)
            .fillna(1)
        )

    return df


def get_win_probs(
    df,
    model,
    calibrator,
    feature_cols
):

    cols = [
        c
        for c in feature_cols
        if c in df.columns
    ]

    X = df[cols].astype(float)

    raw_probs = (
        model.predict_proba(X)[:, 1]
    )

    cal_probs = (
        calibrator.predict(raw_probs)
    )

    df = df.copy()

    df["win_prob"] = cal_probs

    return df


def label_shift(row):

    reasons = []

    if (
        "is_wicket" in row.index
        and row["is_wicket"] == 1
    ):
        reasons.append(
            "Wicket fell"
        )

    if row.get("runs_off_bat", 0) == 6:
        reasons.append("Six hit")

    elif row.get("runs_off_bat", 0) == 4:
        reasons.append("Four hit")

    rrr = row.get(
        "required_run_rate",
        0
    )

    if rrr > 15:

        reasons.append(
            "Required rate > 15"
        )

    elif rrr > 12:

        reasons.append(
            "Required rate > 12"
        )

    over = row.get(
        "over",
        0
    )

    if over == 5:
        reasons.append(
            "Powerplay ended"
        )

    elif over == 15:
        reasons.append(
            "Death overs began"
        )

    wickets_left = row.get(
        "wickets_in_hand",
        10
    )

    if wickets_left <= 2:

        reasons.append(
            "Last pair batting"
        )

    elif wickets_left <= 4:

        reasons.append(
            "Tail-end batting"
        )

    if (
        row.get(
            "dots_last_6",
            0
        ) >= 5
    ):

        reasons.append(
            "5+ dots in last over"
        )

    if (
        row.get(
            "boundaries_last_6",
            0
        ) >= 3
    ):

        reasons.append(
            "3+ boundaries in last over"
        )

    if len(reasons) == 0:

        reasons.append(
            "Bowling/batting pressure shift"
        )

    return " + ".join(reasons)


#---Momentum Shift Detector-----

def detect_momentum_shifts(
    match_df,
    threshold=None
):

    match_df = (
        match_df
        .sort_values(
            ["over", "ball"]
        )
        .reset_index(drop=True)
    )

    match_df["prob_change"] = (
        match_df["win_prob"]
        .diff()
        .fillna(0)
    )

    match_df["abs_change"] = (
        match_df["prob_change"]
        .abs()
    )

    if threshold is None:

        threshold = (
            match_df["abs_change"]
            .quantile(0.90)
        )

    shifts = match_df[
        match_df["abs_change"]
        >= threshold
    ].copy()

    shifts["shift_context"] = (
        shifts.apply(
            label_shift,
            axis=1
        )
    )

    shifts["shift_direction"] = (
        shifts["prob_change"]
        .apply(
            lambda x:
            "positive"
            if x > 0
            else "negative"
        )
    )

    return shifts


def detect_all_matches(df):

    all_shifts = []

    df = df.copy()

    df["prob_change"] = (
        df.groupby(
            "match_id"
        )["win_prob"]
        .diff()
        .fillna(0)
    )

    df["abs_change"] = (
        df["prob_change"]
        .abs()
    )

    global_threshold = (
        df["abs_change"]
        .quantile(0.95)
    )

    print(
        f"Global momentum threshold: "
        f"{global_threshold:.4f}"
    )

    for (
        match_id,
        group
    ) in df.groupby(
        "match_id"
    ):

        shifts = (
            detect_momentum_shifts(
                group,
                threshold=global_threshold
            )
        )

        if len(shifts) > 0:

            shifts["match_id"] = (
                match_id
            )

            all_shifts.append(
                shifts
            )

    if not all_shifts:

        print(
            "No momentum events found"
        )

        return pd.DataFrame()

    result = pd.concat(
        all_shifts,
        ignore_index=True
    )

    print(
        f"Total momentum events: "
        f"{len(result):,}"
    )

    print(
        f"Across "
        f"{result['match_id'].nunique():,} "
        f"matches"
    )

    return result



def build_shap_explainer(
    model,
    X_sample: pd.DataFrame
):
    """
    Build SHAP TreeExplainer.

    Parameters
    ----------
    model : trained XGBoost model
    X_sample : representative sample
               (kept for future use)

    Returns
    -------
    shap.TreeExplainer
    """

    explainer = shap.TreeExplainer(
        model
    )

    return explainer


def explain_momentum_shift(
    explainer,
    ball_row: pd.Series,
    feature_cols: list,
    top_n: int = 3
) -> dict:
    """
    Compute SHAP explanation for a single ball.

    Returns:
    --------
    {
        top_factors,
        explanation,
        shap_values
    }
    """

    cols = [
        c
        for c in feature_cols
        if c in ball_row.index
    ]

    X_df = pd.DataFrame(
    [ball_row[cols]]
)

    X_df = X_df.apply(
    pd.to_numeric,
    errors="coerce"
)

    X_df = X_df.fillna(0)


    shap_values = explainer(X_df)

    if isinstance(
        shap_values,
        list
    ):
        sv = shap_values[0][0]
    else:
        sv = shap_values.values[0]

    shap_series = pd.Series(
        sv,
        index=cols
    )

    excluded = [
    "required_run_rate",
    "required_runs",
    "target",
    "cum_runs",
    "cum_wickets"
    ]

    shap_series = shap_series.drop(
    excluded,
    errors="ignore"
    )

    factors = []

    top_features = (
        shap_series.abs()
        .nlargest(top_n)
    )

    for feat in top_features.index:

        direction = (
            "↑"
            if shap_series[feat] > 0
            else "↓"
        )

        factors.append(
            {
                "feature": feat,
                "shap_value":
                    shap_series[feat],
                "direction":
                    direction,
                "magnitude":
                    abs(
                        shap_series[feat]
                    )
            }
        )

    parts = []

    for factor in factors:

        feat_name = (
            factor["feature"]
            .replace("_", " ")
            .title()
        )

        parts.append(
            f"{feat_name} "
            f"{factor['direction']} "
            f"({factor['shap_value']:+.3f})"
        )

    explanation = (
        " | ".join(parts)
    )

    return {
        "top_factors": factors,
        "explanation": explanation,
        "shap_values": shap_series
    }


def attach_shap_to_shifts(
    shifts_df: pd.DataFrame,
    df_full: pd.DataFrame,
    explainer,
    feature_cols: list
) -> pd.DataFrame:
    """
    Attach SHAP explanations to every
    detected momentum event.
    """

    explanations = []

    print(
        "Computing SHAP for "
        "momentum shifts..."
    )

    for _, row in (
        shifts_df.iterrows()
    ):

        try:

            result = (
                explain_momentum_shift(
                    explainer,
                    row,
                    feature_cols
                )
            )

            explanations.append(
                result[
                    "explanation"
                ]
            )

        except Exception as e:
            print(f"SHAP ERROR: {e}")
            explanations.append("Explanation unavailable")

    shifts_df = (
        shifts_df.copy()
    )

    shifts_df[
        "shap_explanation"
    ] = explanations

    return shifts_df



def compute_player_impact(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attribute win probability changes
    to players.
    """

    df = (
        df
        .sort_values(
            ["match_id", "over", "ball"]
        )
        .copy()
    )

    df["prob_change"] = (
        df.groupby("match_id")
        ["win_prob"]
        .diff()
        .fillna(0)
    )

    records = []

    for _, row in df.iterrows():

        match_id = row["match_id"]

        prob_delta = row[
            "prob_change"
        ]

        # Batting impact
        if row.get(
            "runs_off_bat",
            0
        ) > 0:

            records.append(
                {
                    "match_id":
                        match_id,
                    "player":
                        row.get(
                            "striker",
                            "Unknown"
                        ),
                    "role":
                        "batting",
                    "impact":
                        prob_delta,
                    "season":
                        row.get(
                            "season"
                        ),
                    "team":
                        row.get(
                            "batting_team"
                        )
                }
            )

        # Wicket impact
        if (
            "is_wicket" in row.index
            and row["is_wicket"] == 1
        ):

            dismissed = row.get(
                "player_dismissed",
                row.get(
                    "striker"
                )
            )

            if pd.notna(
                dismissed
            ):

                records.append(
                    {
                        "match_id":
                            match_id,
                        "player":
                            dismissed,
                        "role":
                            "dismissed",
                        "impact":
                            prob_delta,
                        "season":
                            row.get(
                                "season"
                            ),
                        "team":
                            row.get(
                                "batting_team"
                            )
                    }
                )

        # Bowling impact
        if (
            row.get(
                "total_runs_ball",
                0
            ) == 0
            or (
                "is_wicket"
                in row.index
                and row[
                    "is_wicket"
                ] == 1
            )
        ):

            records.append(
                {
                    "match_id":
                        match_id,
                    "player":
                        row.get(
                            "bowler",
                            "Unknown"
                        ),
                    "role":
                        "bowling",
                    "impact":
                        -prob_delta,
                    "season":
                        row.get(
                            "season"
                        ),
                    "team":
                        row.get(
                            "bowling_team"
                        )
                }
            )

    impact_df = pd.DataFrame(
        records
    )

    return impact_df


def aggregate_player_impact(
    impact_df: pd.DataFrame,
    min_matches: int = 5
) -> pd.DataFrame:

    agg = (
        impact_df
        .groupby(
            [
                "player",
                "season",
                "role"
            ]
        )
        .agg(
            total_impact=(
                "impact",
                "sum"
            ),
            positive_impact=(
                "impact",
                lambda x:
                x[x > 0].sum()
            ),
            negative_impact=(
                "impact",
                lambda x:
                x[x < 0].sum()
            ),
            balls_involved=(
                "impact",
                "count"
            ),
            matches=(
                "match_id",
                "nunique"
            )
        )
        .reset_index()
    )

    agg = agg[
        agg["matches"]
        >= min_matches
    ]

    agg[
        "impact_per_ball"
    ] = (
        agg[
            "total_impact"
        ]
        /
        agg[
            "balls_involved"
        ]
    )

    return (
        agg
        .sort_values(
            "total_impact",
            ascending=False
        )
    )


def top_players_by_season(
    agg_df: pd.DataFrame,
    season: int,
    role: str = "batting",
    top_n: int = 10
) -> pd.DataFrame:

    return (
        agg_df[
            (
                agg_df[
                    "season"
                ]
                == season
            )
            &
            (
                agg_df[
                    "role"
                ]
                == role
            )
        ]
        .nlargest(
            top_n,
            "total_impact"
        )
        [
            [
                "player",
                "total_impact",
                "impact_per_ball",
                "matches",
                "balls_involved"
            ]
        ]
    )



def plot_momentum_match(
    match_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    match_id: str
):
    """
    Plot win probability curve with momentum shift markers
    and context labels.
    """

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        range(len(match_df)),
        match_df["win_prob"],
        color="steelblue",
        linewidth=2,
        label="Win Probability"
    )

    ax.fill_between(
        range(len(match_df)),
        match_df["win_prob"],
        alpha=0.15,
        color="steelblue"
    )

    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        alpha=0.5
    )

    match_shifts = (
    shifts_df[
        shifts_df["match_id"].astype(str)
        == str(match_id)
    ]
    .nlargest(8, "abs_change")
    )

    for _, shift in match_shifts.iterrows():

        ball_idx = match_df[
            (
                match_df["over"]
                == shift["over"]
            )
            &
            (
                match_df["ball"]
                == shift["ball"]
            )
        ].index

        if len(ball_idx) == 0:
            continue

        idx = (
            ball_idx[0]
            - match_df.index[0]
        )

        color = (
            "green"
            if shift["shift_direction"]
            == "positive"
            else "red"
        )

        ax.axvline(
            idx,
            color=color,
            alpha=0.4,
            linestyle="--"
        )

        ax.annotate(
            shift["shift_context"][:30],
            xy=(
                idx,
                shift["win_prob"]
            ),
            xytext=(
                idx + 2,
                shift["win_prob"] + 0.05
            ),
            fontsize=7,
            color=color,
            rotation=45
        )

    match_info = match_df.iloc[0]

    ax.set_title(
        f"Match {match_id} | "
        f"{match_info.get('batting_team', '')} chasing | "
        f"Winner: {match_info.get('winner', '')}",
        fontsize=12
    )

    ax.set_xlabel(
        "Ball Number"
    )

    ax.set_ylabel(
        "Chasing Team Win Probability"
    )

    ax.set_ylim(0, 1)

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"models/momentum_{match_id}.png",
        dpi=150
    )

    plt.show()


def plot_player_impact_comparison(
    agg_df: pd.DataFrame,
    player1: str,
    player2: str
):
    """
    Compare two players'
    impact scores across seasons.
    """

    p1 = (
        agg_df[
            (
                agg_df["player"]
                == player1
            )
            &
            (
                agg_df["role"]
                == "batting"
            )
        ]
        .set_index("season")
        ["total_impact"]
    )

    p2 = (
        agg_df[
            (
                agg_df["player"]
                == player2
            )
            &
            (
                agg_df["role"]
                == "batting"
            )
        ]
        .set_index("season")
        ["total_impact"]
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    seasons = sorted(
        set(p1.index)
        | set(p2.index)
    )

    x = range(
        len(seasons)
    )

    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        [
            p1.get(s, 0)
            for s in seasons
        ],
        width,
        label=player1,
        color="steelblue"
    )

    ax.bar(
        [i + width / 2 for i in x],
        [
            p2.get(s, 0)
            for s in seasons
        ],
        width,
        label=player2,
        color="coral"
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        seasons,
        rotation=45
    )

    ax.set_title(
        f"Player Impact Score: "
        f"{player1} vs {player2}"
    )

    ax.set_ylabel(
        "Total Impact Score"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    safe_p1 = (
        player1
        .replace(" ", "_")
    )

    safe_p2 = (
        player2
        .replace(" ", "_")
    )

    plt.savefig(
        f"models/impact_{safe_p1}_vs_{safe_p2}.png",
        dpi=150
    )

    plt.show()




def run_pipeline():

    print("=" * 50)
    print("LOADING ARTIFACTS")
    print("=" * 50)

    model, calibrator, feature_cols = load_artifacts()

    df = load_data()

    print("\n" + "=" * 50)
    print("ATTACHING WIN PROBABILITIES")
    print("=" * 50)

    df = get_win_probs(
        df,
        model,
        calibrator,
        feature_cols
    )

    print("\n" + "=" * 50)
    print("DETECTING MOMENTUM SHIFTS")
    print("=" * 50)

    shifts_df = detect_all_matches(df)

    if len(shifts_df) == 0:

        print("No momentum shifts found.")

        return (
            df,
            pd.DataFrame(),
            pd.DataFrame(),
            None
        )

    print("\n" + "=" * 50)
    print("BUILDING SHAP EXPLAINER")
    print("=" * 50)

    cols = [
        c
        for c in feature_cols
        if c in df.columns
    ]

    sample_size = min(
        500,
        len(df)
    )

    sample = (
        df[cols]
        .astype(float)
        .sample(
            sample_size,
            random_state=42
        )
    )

    explainer = build_shap_explainer(
        model,
        sample
    )

    print(
        "Attaching SHAP explanations..."
    )

    shifts_df = attach_shap_to_shifts(
        shifts_df,
        df,
        explainer,
        feature_cols
    )

    shifts_df.to_parquet(
        PROCESSED_DIR / "momentum_shifts.parquet",
        index=False
    )

    print(
        f"Momentum shifts saved: "
        f"{len(shifts_df):,}"
    )

    print("\nSample momentum events:")

    print(
        shifts_df[
            [
                "match_id",
                "over",
                "ball",
                "prob_change",
                "shift_context",
                "shap_explanation"
            ]
        ]
        .head(5)
        .to_string()
    )

    print("\n" + "=" * 50)
    print("COMPUTING PLAYER IMPACT SCORES")
    print("=" * 50)

    impact_df = compute_player_impact(df)

    agg_df = aggregate_player_impact(
        impact_df
    )

    agg_df.to_parquet(
        PROCESSED_DIR / "player_impact.parquet",
        index=False
    )

    print(
        f"Player impact saved: "
        f"{len(agg_df):,} "
        f"player-season records"
    )

    latest_season = int(
        df["season"].max()
    )

    print(
        f"\nTop 10 Batting Impact ({latest_season})"
    )

    print(
        top_players_by_season(
            agg_df,
            latest_season,
            "batting"
        )
    )

    print(
        f"\nTop 10 Bowling Impact ({latest_season})"
    )

    print(
        top_players_by_season(
            agg_df,
            latest_season,
            "bowling"
        )
    )

    print("\n" + "=" * 50)
    print("PLOTTING MOMENTUM CURVES")
    print("=" * 50)

    top_matches = (
        shifts_df
        .groupby("match_id")
        ["abs_change"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(3)
        .index
    )

    for match_id in top_matches:

        match_df = (
            df[
                df["match_id"]
                == match_id
            ]
            .reset_index(
                drop=True
            )
        )

        plot_momentum_match(
            match_df,
            shifts_df,
            match_id
        )

    print(
        "\nSHAP explainer built "
        "(not saved to disk)"
    )

    print("\n" + "=" * 50)
    print("MATCH INTELLIGENCE PIPELINE COMPLETE")
    print("=" * 50)

    print(
        f"Momentum events: "
        f"{len(shifts_df):,}"
    )

    print(
    "Player Impact Score disabled "
    "(player columns not available)"
)

    print(
        "Momentum shifts file:"
    )

    print(
        PROCESSED_DIR
        / "momentum_shifts.parquet"
    )

    print(
        "Player impact file:"
    )

    print(
        PROCESSED_DIR
        / "player_impact.parquet"
    )

    return (
    df,
    shifts_df,
    agg_df,
    explainer
)


if __name__ == "__main__":

    (
    df,
    shifts_df,
    agg_df,
    explainer
) = run_pipeline()