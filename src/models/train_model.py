import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)

from sklearn.metrics import (
    log_loss,
    brier_score_loss
)

from sklearn.model_selection import (
    train_test_split
)

import matplotlib.pyplot as plt


PROCESSED_DIR = Path("data/processed")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FEATURE_COLS = [

    # Match state
    "over",
    "cum_runs",
    "cum_wickets",
    "wickets_in_hand",
    "balls_remaining",
    "current_run_rate",
    "overs_completed",

    # Chase state
    "target",
    "required_runs",
    "required_run_rate",
    "run_rate_pressure",

    # Momentum
    "runs_last_6",
    "runs_last_18",
    "wickets_last_18",
    "dots_last_6",
    "boundaries_last_6",
    "partnership_runs",
    "partnership_balls",

    # Context
    "venue_avg_score",
]

TARGET_COL = "batting_team_won"



def load_and_prepare(path: Path) -> pd.DataFrame:

    df = pd.read_parquet(path)

    df=df[df["innings"]==2]
    print(f"2nd innings only: {len(df):,} balls")
    phase_map = {
        "powerplay": 0,
        "middle": 1,
        "death": 2
    }

    if "over_phase" in df.columns:

        df["over_phase"] = (
            df["over_phase"]
            .map(phase_map)
            .fillna(1)
            .astype(int)
        )

    if "season" in df.columns:

        df["season"] = (
            df["season"]
            .astype(str)
            .str[:4]
        )

        df["season"] = pd.to_numeric(
            df["season"],
            errors="coerce"
        )

    keep_cols = (
    FEATURE_COLS
    + [
        TARGET_COL,
        "match_id",
        "season",
        "innings",
        "ball",
        "batting_team",
        "bowling_team"
    ]
    )

    keep_cols = [
        c
        for c in keep_cols
        if c in df.columns
    ]

    df = df[keep_cols]
    df = df.loc[:, ~df.columns.duplicated()]

    df = df.dropna(
        subset=[
            c
            for c in FEATURE_COLS
            if c in df.columns
        ]
    )

    df = df.dropna(
        subset=[TARGET_COL]
    )

    print(
        f"Dataset shape after cleaning: {df.shape}"
    )

    print(
        f"Target distribution:\n"
        f"{df[TARGET_COL].value_counts(normalize=True)}"
    )

    return df


def time_based_split(df: pd.DataFrame):

    train = df[
        df["season"] <= 2020
    ]

    val = df[
        df["season"] == 2021
    ]

    test = df[
        df["season"] >= 2022
    ]

    print(
        f"Train: {len(train):,} balls "
        f"({train['match_id'].nunique()} matches)"
    )

    print(
        f"Val: {len(val):,} balls "
        f"({val['match_id'].nunique()} matches)"
    )

    print(
        f"Test: {len(test):,} balls "
        f"({test['match_id'].nunique()} matches)"
    )

    return train, val, test


def get_X_y(df: pd.DataFrame):

    feature_cols = [
        c
        for c in FEATURE_COLS
        if c in df.columns
    ]

    X = (
        df[feature_cols]
        .astype(float)
    )

    y = (
        df[TARGET_COL]
        .astype(int)
    )

    return X, y


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val
):
    """
    Train XGBoost using validation
    log-loss for early stopping.
    """

    model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

    model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
    )

    print(
    f"\nTrees trained: "
    f"{model.n_estimators}"
    )

   

    return model


def evaluate_model(
    model,
    X,
    y,
    split_name: str
):
    """
    Evaluate model performance.
    """

    probs = (
        model.predict_proba(X)
        [:, 1]
    )

    ll = log_loss(
        y,
        probs
    )

    bs = brier_score_loss(
        y,
        probs
    )

    print(
        f"\n{split_name} Results:"
    )

    print(
        f"  Log-loss: "
        f"{ll:.4f}"
    )

    print(
        f"  Brier score: "
        f"{bs:.4f}"
    )

    return (
        probs,
        ll,
        bs
    )

def calibrate_model(
    model,
    X_val,
    y_val
):
    """
    Calibrate probabilities using
    isotonic regression.
    """

    raw_probs = (
        model.predict_proba(X_val)
        [:, 1]
    )

    calibrator = IsotonicRegression(
        out_of_bounds="clip"
    )

    calibrator.fit(
        raw_probs,
        y_val
    )

    calibrated_probs = (
        calibrator.predict(
            raw_probs
        )
    )

    print(
        "\nCalibration Results (Val Set):"
    )

    print(
        f"  Before — Brier: "
        f"{brier_score_loss(y_val, raw_probs):.4f}"
    )

    print(
        f"  After  — Brier: "
        f"{brier_score_loss(y_val, calibrated_probs):.4f}"
    )

    return calibrator


def plot_calibration_curve(
    model,
    calibrator,
    X_val,
    y_val
):
    """
    Plot calibration quality.
    """

    raw_probs = (
        model.predict_proba(X_val)
        [:, 1]
    )

    cal_probs = (
        calibrator.predict(
            raw_probs
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Perfect calibration"
    )

    fraction_pos, mean_pred = (
        calibration_curve(
            y_val,
            raw_probs,
            n_bins=10
        )
    )

    ax.plot(
        mean_pred,
        fraction_pos,
        "s-",
        label="XGBoost (raw)"
    )

    fraction_pos_cal, mean_pred_cal = (
        calibration_curve(
            y_val,
            cal_probs,
            n_bins=10
        )
    )

    ax.plot(
        mean_pred_cal,
        fraction_pos_cal,
        "s-",
        label="XGBoost (calibrated)"
    )

    ax.set_xlabel(
        "Mean Predicted Probability"
    )

    ax.set_ylabel(
        "Actual Win Rate"
    )

    ax.set_title(
        "Calibration Curve"
    )

    ax.legend()

    ax.grid(True)

    plt.tight_layout()

plt.savefig(
    MODELS_DIR / "calibration_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved -> models/calibration_curve.png"
)


def plot_feature_importance(
    model,
    feature_cols
):
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(
        ascending=True
    )

    plt.figure(
        figsize=(10, 8)
    )

    importance.plot(
        kind="barh"
    )

    plt.title(
        "XGBoost Feature Importance"
    )

    plt.xlabel(
        "Importance Score"
    )

    plt.tight_layout()

    plt.savefig(
        MODELS_DIR / "feature_importance.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved -> models/feature_importance.png"
    )


def plot_win_probability_curve(
    model,
    calibrator,
    df_full: pd.DataFrame,
    match_id: str,
    title: str = None
):
    """
    Generate ball-by-ball win probability curve
    for a single match.
    """

    match = df_full[
        df_full["match_id"].astype(str)
        == str(match_id)
    ].copy()

    if match.empty:
        print(f"Match {match_id} not found")
        return

    match = match.sort_values(
        ["innings", "ball"]
    )

    phase_map = {
        "powerplay": 0,
        "middle": 1,
        "death": 2
    }

    if "over_phase" in match.columns:

        match["over_phase"] = (
            match["over_phase"]
            .map(phase_map)
            .fillna(1)
            .astype(int)
        )

    feature_cols = [
        c
        for c in FEATURE_COLS
        if c in match.columns
    ]

    match = match.dropna(
        subset=feature_cols
    )

    X_match = (
        match[feature_cols]
        .astype(float)
    )

    raw_probs = (
        model.predict_proba(X_match)
        [:, 1]
    )

    cal_probs = (
        calibrator.predict(
            raw_probs
        )
    )

    match = match.copy()

    match["win_prob"] = cal_probs

    inn1 = (
        match[match["innings"] == 1]
        .reset_index(drop=True)
    )

    inn2 = (
        match[match["innings"] == 2]
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8)
    )

    team1 = (
        inn1["batting_team"].iloc[0]
        if len(inn1) > 0
        else "Team 1"
    )

    team2 = (
        inn2["batting_team"].iloc[0]
        if len(inn2) > 0
        else "Team 2"
    )

    winner = (
        match["winner"].iloc[0]
        if "winner" in match.columns
        else "Unknown"
    )

    for ax, inn_df, inn_num, team in [
        (axes[0], inn1, 1, team1),
        (axes[1], inn2, 2, team2)
    ]:

        if inn_df.empty:
            continue

        ax.plot(
            range(len(inn_df)),
            inn_df["win_prob"],
            linewidth=2
        )

        ax.fill_between(
            range(len(inn_df)),
            inn_df["win_prob"],
            alpha=0.20
        )

        ax.axhline(
            0.50,
            linestyle="--",
            alpha=0.50
        )

        

        ax.set_ylim(0, 1)

        ax.set_ylabel(
            "Win Probability"
        )

        ax.set_title(
            f"Innings {inn_num} - {team}"
        )

        ax.grid(
            True,
            alpha=0.30
        )

        

    match_title = (
        title
        or f"Match {match_id} | Winner: {winner}"
    )

    fig.suptitle(
        match_title,
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        MODELS_DIR / f"win_prob_{match_id}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved -> models/win_prob_{match_id}.png"
    )




def train_pipeline():

    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)

    df = load_and_prepare(
        PROCESSED_DIR / "features.parquet"
    )

    print("\n" + "=" * 50)
    print("SPLITTING DATA")
    print("=" * 50)

    train, val, test = time_based_split(df)

    if len(val) == 0:
        raise ValueError(
            "Validation set is empty. Check season split."
        )

    if len(test) == 0:
        raise ValueError(
            "Test set is empty. Check season split."
        )

    feature_cols = [
        c
        for c in FEATURE_COLS
        if c in df.columns
    ]

    X_train, y_train = get_X_y(train)
    X_val, y_val = get_X_y(val)
    X_test, y_test = get_X_y(test)

    print("\n" + "=" * 50)
    print("TRAINING XGBOOST")
    print("=" * 50)

    print("\nFEATURE DTYPES")
    print(X_train.dtypes)

    print("\nNON-NUMERIC COLUMNS")
    print(  
    X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()
    )

    model = train_xgboost(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("\n" + "=" * 50)
    print("EVALUATING")
    print("=" * 50)

    _, train_ll, train_bs = evaluate_model(
        model,
        X_train,
        y_train,
        "Train"
    )

    _, val_ll, val_bs = evaluate_model(
        model,
        X_val,
        y_val,
        "Validation"
    )

    _, test_ll, test_bs = evaluate_model(
        model,
        X_test,
        y_test,
        "Test"
    )

    print("\n" + "=" * 50)
    print("CALIBRATING")
    print("=" * 50)

    calibrator = calibrate_model(
        model,
        X_val,
        y_val
    )

    print("\n" + "=" * 50)
    print("PLOTTING CALIBRATION CURVE")
    print("=" * 50)

    plot_calibration_curve(
        model,
        calibrator,
        X_val,
        y_val
    )

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)

    plot_feature_importance(
        model,
        feature_cols
    )

    print("\n" + "=" * 50)
    print("SAVING MODEL")
    print("=" * 50)

    with open(
        MODELS_DIR / "xgb_model.pkl",
        "wb"
    ) as f:
        pickle.dump(model, f)

    with open(
        MODELS_DIR / "calibrator.pkl",
        "wb"
    ) as f:
        pickle.dump(calibrator, f)

    with open(
        MODELS_DIR / "feature_cols.pkl",
        "wb"
    ) as f:
        pickle.dump(feature_cols, f)

    print(
        "\nSaved:"
    )
    print(
        "  models/xgb_model.pkl"
    )
    print(
        "  models/calibrator.pkl"
    )
    print(
        "  models/feature_cols.pkl"
    )

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)

    print(
        f"Train log-loss: {train_ll:.4f}"
    )

    print(
        f"Val   log-loss: {val_ll:.4f}"
    )

    print(
        f"Test  log-loss: {test_ll:.4f}"
    )

    print(
        f"Test  Brier:    {test_bs:.4f}"
    )

    return model, calibrator, df


if __name__ == "__main__":

    model, calibrator, df = train_pipeline()

    print("\nGenerating sample win probability curves...")

    sample_matches = (
        df.groupby("match_id")
        .size()
        .sort_values(ascending=False)
        .index[:3]
    )

    for match_id in sample_matches:

        plot_win_probability_curve(
            model,
            calibrator,
            df,
            match_id
        )