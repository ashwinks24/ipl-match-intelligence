import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw/ipl_csv2")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Pune Warriors": "Rising Pune Supergiant",
    "Pune Warriors India": "Rising Pune Supergiant",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
}


def normalize_team_name(name: str) -> str:
    if pd.isna(name):
        return name

    return TEAM_NAME_MAP.get(
        str(name).strip(),
        str(name).strip()
    )


def load_ball_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:

    ball_files = [
        f for f in raw_dir.glob("*.csv")
        if f.stem.isdigit()
    ]

    print(f"Found {len(ball_files)} ball files")

    dfs = []

    for f in tqdm(ball_files, desc="Loading ball data"):

        try:
            df = pd.read_csv(f)

            required_cols = [
                "match_id",
                "innings",
                "ball"
            ]

            missing = [
                c for c in required_cols
                if c not in df.columns
            ]

            if missing:
                print(f"Skipping {f.name}: {missing}")
                continue

            df["match_id"] = df["match_id"].astype(str)

            dfs.append(df)

        except Exception as e:
            print(f"Failed: {f.name} - {e}")

    if not dfs:
        raise ValueError("No valid files loaded")

    combined = pd.concat(dfs, ignore_index=True)

    before = len(combined)

    combined = combined.drop_duplicates()

    removed = before - len(combined)

    if removed:
        print(f"Removed {removed:,} duplicate rows")

    print(f"Total balls loaded: {len(combined):,}")

    return combined


def load_match_info(raw_dir: Path = RAW_DIR) -> pd.DataFrame:

    info_files = list(raw_dir.glob("*_info.csv"))

    print(f"Found {len(info_files)} info files")

    records = []

    for f in tqdm(info_files, desc="Loading info data"):

        match_id = f.stem.replace("_info", "")

        try:
            df = pd.read_csv(
                f,
                header=None,
                names=["type", "key", "value"],
                engine="python",
                on_bad_lines="skip"
            )   

            info = {
                "match_id": str(match_id)
            }

            for _, row in df.iterrows():

                if row["type"] != "info":
                    continue

                key = (
                    str(row["key"]).strip()
                    if pd.notna(row["key"])
                    else None
                )

                value = (
                    str(row["value"]).strip()
                    if pd.notna(row["value"])
                    else None
                )

                if key:
                    info[key] = value

            records.append(info)

        except Exception as e:
            print(f"Failed: {f.name} - {e}")

    match_df = pd.DataFrame(records)

    print(f"Loaded metadata for {len(match_df)} matches")

    return match_df


def clean_ball_data(df: pd.DataFrame) -> pd.DataFrame:

    for col in ["batting_team", "bowling_team"]:

        if col in df.columns:
            df[col] = df[col].apply(normalize_team_name)

    if "innings" in df.columns:

        before = len(df)

        df = df[
            ~df["innings"].astype(str).str.contains(
                "super",
                case=False,
                na=False
            )
        ]

        print(f"Removed {before - len(df):,} super over balls")

    df["innings"] = pd.to_numeric(
        df["innings"],
        errors="coerce"
    )

    df = df[df["innings"].isin([1, 2])]
    if "season_x" in df.columns:

        df["season_x"] = (
            df["season_x"]
            .astype(str)
            .str[:4]
            .astype(int)
        )

    numeric_cols = [
        "runs_off_bat",
        "extras",
        "wides",
        "noballs",
        "byes",
        "legbyes",
        "penalty"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

        else:
            df[col] = 0

    df["total_runs_ball"] = (
        df["runs_off_bat"] + df["extras"]
    )

    if "wicket_type" in df.columns:

        df["is_wicket"] = (
            df["wicket_type"]
            .notna()
            .astype(int)
        )

    else:
        df["is_wicket"] = 0

    df["is_legal_ball"] = (
        (df["wides"] == 0)
        & (df["noballs"] == 0)
    ).astype(int)

    df["is_four"] = (
        df["runs_off_bat"] == 4
    ).astype(int)

    df["is_six"] = (
        df["runs_off_bat"] == 6
    ).astype(int)

    return df


def add_cumulative_features(
    df: pd.DataFrame,
    total_legal_balls: int = 120
) -> pd.DataFrame:

    df = df.copy()

    df["ball"] = pd.to_numeric(
        df["ball"],
        errors="coerce"
    )

    df = df.sort_values(
        ["match_id", "innings", "ball"]
    )

    group_cols = ["match_id", "innings"]

    g = df.groupby(group_cols)

    df["cum_runs"] = (
        g["total_runs_ball"]
        .cumsum()
    )

    df["cum_legal_balls"] = (
        g["is_legal_ball"]
        .cumsum()
    )

    df["cum_wickets_after"] = (
        g["is_wicket"]
        .cumsum()
    )

    df["cum_wickets"] = (
        df["cum_wickets_after"]
        - df["is_wicket"]
    )

    df["overs_completed"] = (
        df["cum_legal_balls"] / 6
    )

    df["balls_remaining"] = (
        total_legal_balls
        - df["cum_legal_balls"]
    ).clip(lower=0)

    df["current_run_rate"] = np.where(
        df["overs_completed"] > 0,
        df["cum_runs"] / df["overs_completed"],
        0
    )

    return df


def merge_info(
    ball_df: pd.DataFrame,
    match_df: pd.DataFrame
) -> pd.DataFrame:

    merged = ball_df.merge(
        match_df,
        on="match_id",
        how="left"
    )

    unmatched = (
        merged["match_id"][merged["venue"].isna()]
        .nunique()
        if "venue" in merged.columns
        else 0
    )

    total_matches = ball_df["match_id"].nunique()

    merge_rate = (
        1 - (unmatched / total_matches)
        if total_matches > 0 else 0
    )

    print(
        f"Merge success rate: "
        f"{merge_rate:.1%}"
    )

    return merged


def load_and_prepare_data(
    raw_dir: Path = RAW_DIR,
    save: bool = True,
    total_legal_balls: int = 120
) -> pd.DataFrame:

    print("=" * 60)
    print("STEP 1: Loading ball data")

    ball_df = load_ball_data(raw_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Loading match info")

    match_df = load_match_info(raw_dir)

    print("\n" + "=" * 60)
    print("STEP 3: Cleaning data")

    ball_df = clean_ball_data(ball_df)

    print("\n" + "=" * 60)
    print("STEP 4: Merging datasets")

    df = merge_info(ball_df, match_df)

    print("\n" + "=" * 60)
    print("STEP 5: Adding cumulative features")

    df = add_cumulative_features(
        df,
        total_legal_balls=total_legal_balls
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")

    print(f"Total balls:   {len(df):,}")
    print(f"Total matches: {df['match_id'].nunique():,}")

    if "season" in df.columns:
        print(
            f"Seasons: "
            f"{sorted(df['season'].dropna().unique())}"
        )

    print(f"Total columns: {len(df.columns)}")
    for col in df.columns:

        if "season" in col:
            df[col] = df[col].astype(str)


    if save:

        out_path = (
            PROCESSED_DIR
            / "ball_by_ball.parquet"
        )

        df.to_parquet(
            out_path,
            index=False
        )

        print(f"\nSaved -> {out_path}")

    return df


if __name__ == "__main__":

    df = load_and_prepare_data()

    print("\nDataset Preview:\n")

    print(df.head())