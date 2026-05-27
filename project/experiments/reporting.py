import glob
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_log_files(log_file=None, log_pattern="logs/games_worker_*.jsonl", worker_id=None):
    if worker_id:
        candidate = BASE_DIR / "logs" / f"games_worker_{worker_id}.jsonl"
        return [candidate] if candidate.exists() else []

    if log_file:
        candidate = Path(log_file)
        if candidate.is_file():
            return [candidate.resolve()]
        if not candidate.is_absolute():
            project_candidate = BASE_DIR / candidate
            if project_candidate.is_file():
                return [project_candidate.resolve()]
        raise FileNotFoundError(f"Log file not found: {candidate}")

    pattern = Path(log_pattern)
    if not pattern.is_absolute():
        pattern = BASE_DIR / pattern
    return [Path(path) for path in sorted(glob.glob(str(pattern)))]


def _infer_bandit_type(path, row):
    if row.get("bandit_type"):
        return row["bandit_type"]
    name = Path(path).name
    if "_neural" in name:
        return "neural_linucb"
    return "basic_linucb"


def load_training_logs(
    log_file=None,
    log_pattern="logs/games_worker_*.jsonl",
    worker_id=None,
    bandit_type=None,
):
    rows = []
    for path in _resolve_log_files(
        log_file=log_file,
        log_pattern=log_pattern,
        worker_id=worker_id,
    ):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_bandit_type = _infer_bandit_type(path, row)
                if bandit_type and row_bandit_type != bandit_type:
                    continue
                row["bandit_type"] = row_bandit_type
                row["source_file"] = str(path)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_training_logs(df):
    if df.empty:
        return {"stats": None}

    df = df.copy()
    for column in ["reward", "elapsed", "ply", "arm", "stockfish_level", "opponent_stockfish_level"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    arm_counts = (
        df["arm"].dropna().astype(int).value_counts().sort_index().to_dict()
        if "arm" in df.columns else {}
    )

    if "game" in df.columns:
        reward_groups = ["source_file", "game"] if "source_file" in df.columns else ["game"]
        recent_rewards = (
            df.groupby(reward_groups, sort=False)["reward"]
            .mean()
            .tail(200)
            .round(6)
            .tolist()
        )
    else:
        recent_rewards = df["reward"].tail(200).round(6).tolist()

    level_column = (
        "opponent_stockfish_level"
        if "opponent_stockfish_level" in df.columns
        else "stockfish_level"
    )
    level_stats = {}
    if level_column in df.columns:
        for level, group in df.dropna(subset=[level_column]).groupby(level_column):
            if "game" in group.columns:
                rewards = group.groupby("game", sort=False)["reward"].mean().tail(200).round(6).tolist()
            else:
                rewards = group["reward"].tail(200).round(6).tolist()
            level_stats[str(int(level))] = {
                "avg_reward": round(float(group["reward"].mean()), 3),
                "rewards": rewards,
            }

    phase_stats = {}
    if "ply" in df.columns and "elapsed" in df.columns:
        df_phase = df.dropna(subset=["ply", "elapsed"]).copy()
        df_phase["move_number"] = (df_phase["ply"] // 2).astype(int)
        df_phase["phase_jeu"] = (df_phase["move_number"] // 5) * 5
        phase_data = df_phase.groupby("phase_jeu")["elapsed"].mean()
        phase_stats = {int(k): round(float(v), 2) for k, v in phase_data.items()}

    arm_time_stats = {}
    if "arm" in df.columns and "elapsed" in df.columns:
        arm_time_df = df.dropna(subset=["arm", "elapsed"])
        if not arm_time_df.empty:
            arm_time_data = arm_time_df.groupby("arm")["elapsed"].mean()
            arm_time_stats = {str(int(k)): round(float(v), 2) for k, v in arm_time_data.items()}

    stats = {
        "rows": int(len(df)),
        "games": int(df["game"].nunique()) if "game" in df.columns else 0,
        "bandit_types": sorted(df["bandit_type"].dropna().unique().tolist()),
    }
    return {
        "stats": stats,
        "arm_counts": arm_counts,
        "recent_rewards": recent_rewards,
        "level_stats": level_stats,
        "phase_stats": phase_stats,
        "arm_time_stats": arm_time_stats,
    }


def load_benchmark_results(bandit_type, csv_path=None):
    path = Path(csv_path) if csv_path else BASE_DIR / "logs" / f"benchmark_results_{bandit_type}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
