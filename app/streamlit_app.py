import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
from pathlib import Path

# ── Page config----
st.set_page_config(
    page_title="IPL Match Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ----
MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

PHASE_MAP = {"powerplay": 0, "middle": 1, "death": 2}

PRIMARY_COLOR = "#1565C0"      # Deep blue
POSITIVE_COLOR = "#2E7D32"     # Professional green
NEGATIVE_COLOR = "#C62828"     # Professional red
BACKGROUND_COLOR = "#FAFAFA"
GRID_COLOR = "#D6D6D6"

FEATURE_COLS = [
    "over",
    "cum_runs",
    "cum_wickets",
    "wickets_in_hand",
    "balls_remaining",
    "current_run_rate",
    "overs_completed",
    "target",
    "required_runs",
    "required_run_rate",
    "run_rate_pressure",
    "runs_last_6",
    "runs_last_18",
    "wickets_last_18",
    "dots_last_6",
    "boundaries_last_6",
    "partnership_runs",
    "partnership_balls",
    "venue_avg_score",
]


# ── Load artifacts (cached)----
@st.cache_resource
def load_model():
    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, calibrator, feature_cols


@st.cache_data
def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    shifts = pd.read_parquet(PROCESSED_DIR / "momentum_shifts.parquet")
    impact = pd.read_parquet(PROCESSED_DIR / "player_impact.parquet")

    shifts["abs_change"] = shifts["prob_change"].abs()

    return df, shifts, impact


@st.cache_data
def get_win_probs_cached(_model, _calibrator, match_id, _df):
    """Compute win probs for one match — cached per match_id."""
    match = _df[_df["match_id"].astype(str) == str(match_id)].copy()
    match = match[match["innings"] == 2].copy()

    if match.empty:
        return match

    if "over_phase" in match.columns:
        match["over_phase"] = match["over_phase"].map(PHASE_MAP).fillna(1)

    cols = [
    c for c in _model.get_booster().feature_names
    if c in match.columns
    ]
    X = match[cols].astype(float)

    raw = _model.predict_proba(X)[:, 1]
    match["win_prob"] = _calibrator.predict(raw)

    return match.reset_index(drop=True)


# ── Sidebar -----
def render_sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/en/8/84/"
        "Indian_Premier_League_Official_Logo.svg",
        width=120
    )
    st.sidebar.title("🏏 IPL Intelligence")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Ball-by-ball win probability engine with "
        "momentum shift detection and SHAP explainability."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Built by Ashwin Kumar Singh**\n\n"
        "[GitHub](https://github.com/ashwinks24/ipl-match-intelligence)"
    )


# ── Tab 1: Match Analyzer----
def render_match_analyzer(df, shifts, model, calibrator, feature_cols):
    st.header("🎯 Match Analyzer")
    st.markdown(
        "Select any IPL match to see ball-by-ball win probability "
        "with momentum shift explanations."
    )

    col1, col2 = st.columns(2)

    with col1:
        seasons = sorted(df["season"].dropna().unique(), reverse=True)
        selected_season = st.selectbox("Select Season", seasons)

    # Filter matches for selected season
    season_df = df[df["season"] == selected_season]
    matches = season_df.groupby("match_id").agg(
        team1=("batting_team", "first"),
        team2=("bowling_team", "first"),
        date=("start_date", "first"),
        winner=("winner", "first")
    ).reset_index()

    matches["label"] = (
        matches["team1"] + " vs " +
        matches["team2"] + " | " +
        matches["date"].astype(str).str[:10]
    )

    with col2:
        selected_label = st.selectbox(
            "Select Match", matches["label"].tolist()
        )

    selected_match = matches[matches["label"] == selected_label].iloc[0]
    match_id = selected_match["match_id"]
    winner = selected_match["winner"]

    # Compute win probs
    with st.spinner("Computing win probability..."):
        match_df = get_win_probs_cached(
            model, calibrator, match_id, df
        )

    if match_df.empty:
        st.warning("No 2nd innings data for this match.")
        return

    chasing_team = match_df["batting_team"].iloc[0]
    target = int(match_df["target"].iloc[0])

    # ── Match summary ──
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chasing Team", chasing_team)
    m2.metric("Target", target)
    m3.metric("Winner", winner)
    m4.metric(
        "Result",
        "Chase Successful" if winner == chasing_team else "Defended"
    )

    # ── Win probability chart ──
    st.markdown("### Win Probability Curve")

    fig = go.Figure()

    # Main probability line
    fig.add_trace(go.Scatter(
        x=list(range(len(match_df))),
        y=match_df["win_prob"],
        mode="lines",
        name=f"{chasing_team} win probability",
        
        fill="tozeroy",
        line=dict(color=PRIMARY_COLOR, width=4),
        fillcolor="rgba(21, 101, 192, 0.25)"
    ))
    
    # 50% line
    fig.add_hline(
    y=0.5,
    line_dash="dash",
    line_color="black",
    opacity=0.8
    )

    # Wicket markers
    wickets = match_df[match_df["is_wicket"] == 1]
    fig.add_trace(go.Scatter(
        x=wickets.index.tolist(),
        y=wickets["win_prob"].tolist(),
        mode="markers",
        name="Wicket",
        marker=dict(
        color=NEGATIVE_COLOR,
        size=14,
        symbol="x",
        line=dict(width=2)
    )
    ))

    # Momentum shift markers
# Momentum shift markers
    match_shifts = shifts[
    shifts["match_id"].astype(str) == str(match_id)
]

    for _, shift in match_shifts.iterrows():

        ball_idx = match_df[
        (match_df["over"] == shift["over"])
        &
        (match_df["ball"] == shift["ball"])
        ].index

        if len(ball_idx) == 0:
            continue

        color = (
        POSITIVE_COLOR
        if shift["prob_change"] > 0
        else NEGATIVE_COLOR
        )

        fig.add_vline(
        x=int(ball_idx[0]),
        line_color=color,
        line_dash="dot",
        opacity=0.4
        )

    fig.update_layout(
    height=500,
    hovermode="x unified",
    plot_bgcolor=BACKGROUND_COLOR,
    paper_bgcolor=BACKGROUND_COLOR,
    template="plotly_white",

    font=dict(
        size=14,
        color="#212121"
    ),

    legend=dict(
        orientation="h",
        y=-0.2,
        font=dict(
            size=14,
            color="#212121"
        )
    ),

    xaxis=dict(
        title="Ball Number",
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        tickfont=dict(
            size=14,
            color="#212121"
        ),
        title_font=dict(
            size=18,
            color="#212121"
        )
    ),

    yaxis=dict(
        title="Win Probability",
        range=[0, 1],
        tickformat=".0%",
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        tickfont=dict(
            size=14,
            color="#212121"
        ),
        title_font=dict(
            size=18,
            color="#212121"
        )
    )
)
    st.plotly_chart(
    fig,
    use_container_width=True
    )
   


    # ── Momentum events table ──
    if len(match_shifts) > 0:
        st.markdown("### 🔄 Momentum Shifts Detected")
        st.markdown(
            f"**{len(match_shifts)} momentum events** found in this match"
        )

        display_cols = [
            "over", "ball", "prob_change",
            "shift_context", "shap_explanation"
        ]
        display_cols = [c for c in display_cols if c in match_shifts.columns]

        display_df = (
            match_shifts[display_cols]
            .copy()
        )

        display_df["abs_change"] = (
        display_df["prob_change"].abs()
        )

        display_df = (
        display_df
        .sort_values(
        "abs_change",
        ascending=False
    )
    .drop(columns=["abs_change"])
)
        
        

        display_df["prob_change"] = display_df["prob_change"].apply(
            lambda x: f"{x:+.1%}"
        )
        display_df.columns = [
            "Over", "Ball", "Prob Change",
            "What Happened", "SHAP Explanation"
        ][:len(display_cols)]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No major momentum shifts detected in this match.")


# ── Tab 2: Player Impact ----
def render_player_impact(impact_df):
    st.header("⚡ Player Impact Score")
    st.markdown(
        "Traditional stats treat all runs equally. "
        "Impact Score weights contributions by match situation — "
        "a boundary in over 19 of a close chase matters more than in over 2."
    )

    col1, col2 = st.columns(2)

    with col1:
        seasons = sorted(
            impact_df["season"].dropna().unique(), reverse=True
        )
        selected_season = st.selectbox(
            "Season", seasons, key="impact_season"
        )

    with col2:
        role = st.selectbox(
            "Role", ["batting", "bowling"], key="impact_role"
        )

    season_impact = impact_df[
        (impact_df["season"] == selected_season) &
        (impact_df["role"] == role)
    ].nlargest(15, "total_impact")

    if season_impact.empty:
        st.warning("No data for selected filters.")
        return

    # ── Top players bar chart ──
    fig = px.bar(
        season_impact,
        x="total_impact",
        y="player",
        orientation="h",
        color="impact_per_ball",
        color_continuous_scale="Blues",
        labels={
            "total_impact": "Total Impact Score",
            "player": "Player",
            "impact_per_ball": "Impact/Ball"
        },
        title=f"Top 15 {role.title()} Impact — IPL {int(selected_season)}"
    )
    fig.update_layout(
        height=500,
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Player comparison ──
    st.markdown("### Compare Two Players")

    all_players = sorted(
        impact_df[impact_df["role"] == role]["player"].unique()
    )

    pc1, pc2 = st.columns(2)
    with pc1:
        p1 = st.selectbox("Player 1", all_players, key="p1")
    with pc2:
        p2 = st.selectbox(
            "Player 2",
            all_players,
            index=min(1, len(all_players)-1),
            key="p2"
        )

    p1_data = impact_df[
        (impact_df["player"] == p1) & (impact_df["role"] == role)
    ].set_index("season")["total_impact"]

    p2_data = impact_df[
        (impact_df["player"] == p2) & (impact_df["role"] == role)
    ].set_index("season")["total_impact"]

    all_seasons = sorted(set(p1_data.index) | set(p2_data.index))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=[str(int(s)) for s in all_seasons],
        y=[p1_data.get(s, 0) for s in all_seasons],
        name=p1,
        marker_color="#1f77b4"
    ))
    fig2.add_trace(go.Bar(
        x=[str(int(s)) for s in all_seasons],
        y=[p2_data.get(s, 0) for s in all_seasons],
        name=p2,
        marker_color="#ff7f0e"
    ))
    fig2.update_layout(
        barmode="group",
        title=f"{p1} vs {p2} — Impact Score by Season",
        xaxis_title="Season",
        yaxis_title="Total Impact Score",
        height=400,
        plot_bgcolor="white"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw table ──
    st.markdown("### Full Impact Table")
    st.dataframe(
        season_impact[[
            "player", "total_impact", "impact_per_ball",
            "matches", "balls_involved"
        ]].round(4),
        use_container_width=True,
        hide_index=True
    )


# ── Tab 3: Season Overview --
def render_season_overview(df, shifts, impact_df):
    st.header("📊 Season Overview")

    seasons = sorted(df["season"].dropna().unique(), reverse=True)
    selected_season = st.selectbox(
        "Select Season", seasons, key="overview_season"
    )

    season_df = df[df["season"] == selected_season]
    season_shifts = shifts[
        shifts["match_id"].isin(
            season_df["match_id"].unique()
        )
    ]

    # ── Key metrics ──
    st.markdown("### Season at a Glance")
    k1, k2, k3, k4 = st.columns(4)

    total_matches = season_df["match_id"].nunique()
    avg_target = season_df[season_df["innings"] == 2]["target"].mean()
    total_momentum = len(season_shifts)
    chase_wins = (
        season_df[season_df["innings"] == 2]
        .groupby("match_id")
        .first()
        .apply(
            lambda x: 1 if x["batting_team"] == x["winner"] else 0,
            axis=1
        )
        .mean()
    )

    k1.metric("Total Matches", total_matches)
    k2.metric("Avg Target", f"{avg_target:.0f}")
    k3.metric("Momentum Events", total_momentum)
    k4.metric("Chase Win Rate", f"{chase_wins:.1%}")

    # ── Most dramatic matches ──
    st.markdown("### Most Dramatic Matches")
    st.markdown("Ranked by total momentum swing magnitude")

    drama = (
        season_shifts.groupby("match_id")["abs_change"]
        .sum()
        .reset_index()
        .rename(columns={"abs_change": "drama_score"})
        .nlargest(10, "drama_score")
    )

    match_info = season_df.groupby("match_id").agg(
        team1=("batting_team", "first"),
        team2=("bowling_team", "first"),
        winner=("winner", "first")
    ).reset_index()

    drama = drama.merge(match_info, on="match_id", how="left")
    drama["match"] = drama["team1"] + " vs " + drama["team2"]
    drama["drama_score"] = drama["drama_score"].round(3)

    st.dataframe(
        drama[["match", "winner", "drama_score"]],
        use_container_width=True,
        hide_index=True
    )

    # ── Venue win rate ──
    st.markdown("### Chase Win Rate by Venue")

    venue_stats = (
        season_df[season_df["innings"] == 2]
        .groupby(["match_id", "venue"])
        .first()
        .reset_index()
    )
    venue_stats["chase_won"] = (
        venue_stats["batting_team"] == venue_stats["winner"]
    ).astype(int)

    venue_agg = (
        venue_stats.groupby("venue")
        .agg(matches=("match_id", "count"),
             chase_win_rate=("chase_won", "mean"))
        .reset_index()
        .query("matches >= 2")
        .sort_values("chase_win_rate", ascending=True)
    )

    fig = px.bar(
        venue_agg,
        x="chase_win_rate",
        y="venue",
        orientation="h",
        color="chase_win_rate",
        color_continuous_scale="RdYlGn",
        title=f"Chase Win Rate by Venue — {int(selected_season)}",
        labels={"chase_win_rate": "Chase Win Rate", "venue": "Venue"}
    )
    fig.update_layout(height=400, plot_bgcolor="white")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


# ── Main App ---
def main():
    render_sidebar()

    st.title("🏏 IPL Match Intelligence System")
    st.markdown(
        "Ball-by-ball win probability • Momentum detection • "
        "SHAP explainability • Player Impact Score"
    )
    st.markdown("---")

    # Load everything
    with st.spinner("Loading models and data..."):
        model, calibrator, feature_cols = load_model()
        df, shifts, impact = load_data()

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Match Analyzer",
        "⚡ Player Impact",
        "📊 Season Overview"
    ])

    with tab1:
        render_match_analyzer(
            df, shifts, model, calibrator, feature_cols
        )

    with tab2:
        render_player_impact(impact)

    with tab3:
        render_season_overview(df, shifts, impact)


if __name__ == "__main__":
    main()