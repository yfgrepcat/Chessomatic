import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Script used to analyse training logs and checkpoint benchmark results.

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.reporting import load_training_logs, summarize_training_logs


ARM_COUNT = 4


def _resolve_output_dir(output_dir):
    if not output_dir:
        return None
    path = Path(output_dir)
    if not path.is_absolute():
        path = BASE_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_or_show(fig, output_dir, filename, show=True):
    fig.tight_layout()
    if output_dir:
        path = output_dir / filename
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"Saved plot: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _prepare_logs(df):
    df = df.copy()
    numeric_columns = [
        "game",
        "ply",
        "arm",
        "reward",
        "move_reward",
        "elapsed",
        "budget",
        "outcome",
        "flag_penalty",
        "final_white_clock",
        "final_black_clock",
        "agent_stockfish_level",
        "opponent_stockfish_level",
        "stockfish_level",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "mab_flagged" in df.columns:
        df["mab_flagged"] = df["mab_flagged"].fillna(False).astype(bool)
    if "opponent_flagged" in df.columns:
        df["opponent_flagged"] = df["opponent_flagged"].fillna(False).astype(bool)
    return df


def _arm_share_by_game(df):
    counts = df.groupby(["game", "arm"]).size().unstack(fill_value=0)
    for arm in range(ARM_COUNT):
        if arm not in counts.columns:
            counts[arm] = 0
    counts = counts[sorted(counts.columns)]
    return counts.div(counts.sum(axis=1), axis=0).fillna(0.0)


def _normalized_entropy(row):
    values = np.asarray(row, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    entropy = float(-(values * np.log(values)).sum() / math.log(ARM_COUNT))
    return max(0.0, min(1.0, entropy))


def _result_score(result):
    if result == "1-0":
        return 1.0
    if result == "1/2-1/2":
        return 0.5
    if result == "0-1":
        return 0.0
    return np.nan


def _per_game_summary(df):
    rows = []
    if "game" not in df.columns:
        return pd.DataFrame(rows)

    for game_id, group in df.groupby("game", sort=True):
        last = group.iloc[-1]
        row = {
            "game": int(game_id),
            "mab_moves": int(len(group)),
            "mean_reward": float(group["reward"].mean()) if "reward" in group else np.nan,
            "mean_move_reward": (
                float(group["move_reward"].mean()) if "move_reward" in group else np.nan
            ),
            "mean_elapsed": float(group["elapsed"].mean()) if "elapsed" in group else np.nan,
            "result": last.get("result", None),
            "score": _result_score(last.get("result", None)),
            "mab_flagged": bool(last.get("mab_flagged", False)),
            "opponent_flagged": bool(last.get("opponent_flagged", False)),
            "final_white_clock": float(last.get("final_white_clock", np.nan)),
            "final_black_clock": float(last.get("final_black_clock", np.nan)),
        }
        for arm in range(ARM_COUNT):
            row[f"arm{arm}_share"] = float((group["arm"] == arm).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _segment_summary(df, games_per_segment):
    per_game = _per_game_summary(df)
    if per_game.empty:
        return pd.DataFrame()

    rows = []
    min_game = int(per_game["game"].min())
    max_game = int(per_game["game"].max())
    for start in range(min_game, max_game + 1, games_per_segment):
        end = start + games_per_segment - 1
        games = per_game[(per_game["game"] >= start) & (per_game["game"] <= end)]
        if games.empty:
            continue
        segment_moves = df[df["game"].isin(games["game"])]
        rows.append({
            "games": f"{start + 1}-{end + 1}",
            "logged_games": int(len(games)),
            "wins": int((games["result"] == "1-0").sum()),
            "draws": int((games["result"] == "1/2-1/2").sum()),
            "losses": int((games["result"] == "0-1").sum()),
            "score_rate": round(float(games["score"].mean()), 3),
            "mab_flag_rate": round(float(games["mab_flagged"].mean()), 3),
            "opponent_flag_rate": round(float(games["opponent_flagged"].mean()), 3),
            "mean_mab_moves": round(float(games["mab_moves"].mean()), 1),
            "mean_final_white_clock": round(float(games["final_white_clock"].mean()), 2),
            "mean_reward": round(float(segment_moves["reward"].mean()), 4),
            "mean_move_reward": (
                round(float(segment_moves["move_reward"].mean()), 4)
                if "move_reward" in segment_moves.columns else np.nan
            ),
            "arm0": round(float((segment_moves["arm"] == 0).mean()), 3),
            "arm1": round(float((segment_moves["arm"] == 1).mean()), 3),
            "arm2": round(float((segment_moves["arm"] == 2).mean()), 3),
            "arm3": round(float((segment_moves["arm"] == 3).mean()), 3),
        })
    return pd.DataFrame(rows)


def _exploration_table(df, window_games):
    arm_share = _arm_share_by_game(df)
    table = pd.DataFrame(index=arm_share.index)
    table["exploration_entropy"] = arm_share.apply(_normalized_entropy, axis=1)
    table["exploitation_concentration"] = 1.0 - table["exploration_entropy"]
    table["max_arm_share"] = arm_share.max(axis=1)
    table["dominant_arm"] = arm_share.idxmax(axis=1)
    table["exploration_entropy_roll"] = (
        table["exploration_entropy"].rolling(window_games, min_periods=1).mean()
    )
    table["exploitation_concentration_roll"] = (
        table["exploitation_concentration"].rolling(window_games, min_periods=1).mean()
    )
    return table.reset_index().rename(columns={"index": "game"})


def _print_tables(df, games_per_segment, window_games):
    summary = summarize_training_logs(df)
    print("\nLoaded rows:")
    print(len(df))
    print("\nShared summary:")
    print(summary["stats"])

    print("\nAverage reward:")
    print(round(float(df["reward"].mean()), 5))

    if "move_reward" in df.columns:
        print("\nAverage move_reward:")
        print(round(float(df["move_reward"].mean()), 5))

    print("\nArm distribution:")
    print(df["arm"].value_counts(normalize=True).sort_index().round(3).to_string())

    print("\nAverage elapsed time by arm:")
    print(df.groupby("arm")["elapsed"].mean().round(3).to_string())

    per_game = _per_game_summary(df)
    if not per_game.empty:
        print("\nPer-game outcome summary:")
        columns = [
            "game",
            "result",
            "score",
            "mab_flagged",
            "opponent_flagged",
            "mab_moves",
            "final_white_clock",
            "mean_reward",
            "mean_move_reward",
        ]
        print(per_game[columns].round(4).to_string(index=False))

    segment = _segment_summary(df, games_per_segment)
    if not segment.empty:
        print(f"\nSegment summary ({games_per_segment} games):")
        print(segment.to_string(index=False))

    exploration = _exploration_table(df, window_games)
    print("\nExploration vs exploitation proxy:")
    print(
        exploration[
            [
                "game",
                "exploration_entropy",
                "exploitation_concentration",
                "max_arm_share",
                "dominant_arm",
            ]
        ].round(3).to_string(index=False)
    )
    print(
        "\nNote: this is a policy-diversity proxy. Exact UCB exploration would require logging "
        "per-arm UCB scores or uncertainty at move time."
    )


def _plot_outcomes(df, output_dir, show):
    per_game = _per_game_summary(df)
    if per_game.empty:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(per_game["game"] + 1, per_game["score"], marker="o", label="Game score")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Training game")
    ax1.set_ylabel("Score (win=1, draw=0.5, loss=0)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(
        per_game["game"] + 1,
        per_game["mab_flagged"].astype(int),
        where="mid",
        color="#d62728",
        alpha=0.5,
        label="MAB flagged",
    )
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Flag")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title("Training outcomes and flags")
    _save_or_show(fig, output_dir, "training_outcomes.png", show=show)


def _plot_rewards(df, output_dir, show, window_games):
    per_game = _per_game_summary(df)
    if per_game.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x = per_game["game"] + 1
    ax.plot(x, per_game["mean_reward"], marker="o", label="Mean total reward")
    if "mean_move_reward" in per_game.columns and per_game["mean_move_reward"].notna().any():
        ax.plot(x, per_game["mean_move_reward"], marker="o", label="Mean move_reward")
    ax.plot(
        x,
        per_game["mean_reward"].rolling(window_games, min_periods=1).mean(),
        linewidth=3,
        alpha=0.55,
        label=f"Reward rolling mean ({window_games} games)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Training game")
    ax.set_ylabel("Reward")
    ax.set_title("Per-game reward diagnostics")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(fig, output_dir, "reward_by_game.png", show=show)


def _plot_arm_usage(df, output_dir, show):
    arm_share = _arm_share_by_game(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    arm_share.index = arm_share.index + 1
    arm_share.plot.area(ax=ax, stacked=True, alpha=0.85)
    ax.set_xlabel("Training game")
    ax.set_ylabel("Arm share")
    ax.set_ylim(0, 1)
    ax.set_title("Arm usage share per game")
    ax.grid(alpha=0.25)
    ax.legend(title="Arm", loc="upper right")
    _save_or_show(fig, output_dir, "arm_usage_share.png", show=show)


def _plot_exploration_vs_exploitation(df, output_dir, show, window_games):
    exploration = _exploration_table(df, window_games)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = exploration["game"] + 1
    ax.plot(
        x,
        exploration["exploration_entropy"],
        marker="o",
        alpha=0.35,
        label="Exploration proxy (arm entropy)",
    )
    ax.plot(
        x,
        exploration["exploitation_concentration"],
        marker="o",
        alpha=0.35,
        label="Exploitation proxy (1 - entropy)",
    )
    ax.plot(
        x,
        exploration["exploration_entropy_roll"],
        linewidth=3,
        label=f"Exploration rolling mean ({window_games} games)",
    )
    ax.plot(
        x,
        exploration["exploitation_concentration_roll"],
        linewidth=3,
        label=f"Exploitation rolling mean ({window_games} games)",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Training game")
    ax.set_ylabel("Proxy value")
    ax.set_title("Exploration vs exploitation proxy")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(fig, output_dir, "exploration_vs_exploitation.png", show=show)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(x, exploration["max_arm_share"], marker="o", color="#9467bd")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Training game")
    ax2.set_ylabel("Max arm share")
    ax2.set_title("Policy concentration by game")
    ax2.grid(alpha=0.3)
    _save_or_show(fig2, output_dir, "policy_concentration.png", show=show)


def _find_checkpoint_csv(worker_id, bandit_type):
    if not worker_id or not bandit_type:
        return None
    candidate = BASE_DIR / "logs" / f"checkpoint_evaluation_{bandit_type}_{worker_id}.csv"
    return candidate if candidate.exists() else None


def _plot_checkpoint_evaluation(checkpoint_csv, output_dir, show):
    if not checkpoint_csv:
        return
    path = Path(checkpoint_csv)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        print(f"\nCheckpoint CSV not found: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    level_column = "opponent_level" if "opponent_level" in df.columns else "level"
    print("\nCheckpoint evaluation:")
    columns = [
        "training_games",
        level_column,
        "wins",
        "losses",
        "draws",
        "winrate",
        "loss_on_time_rate",
        "mean_plies_to_win",
        "mean_clock_used_per_win",
    ]
    columns = [column for column in columns if column in df.columns]
    print(df[columns].to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    for level, group in df.groupby(level_column):
        group = group.sort_values("training_games")
        ax.plot(group["training_games"], group["winrate"], marker="o", label=f"SF {level}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Training games")
    ax.set_ylabel("Winrate")
    ax.set_title("Checkpoint winrate by opponent level")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(fig, output_dir, "checkpoint_winrate.png", show=show)

    if "loss_on_time_rate" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for level, group in df.groupby(level_column):
            group = group.sort_values("training_games")
            ax2.plot(
                group["training_games"],
                group["loss_on_time_rate"],
                marker="o",
                label=f"SF {level}",
            )
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("Training games")
        ax2.set_ylabel("Loss-on-time rate")
        ax2.set_title("Checkpoint loss-on-time rate")
        ax2.legend()
        ax2.grid(alpha=0.3)
        _save_or_show(fig2, output_dir, "checkpoint_loss_on_time.png", show=show)

    if "mean_plies_to_win" in df.columns:
        wins = df[df["mean_plies_to_win"].notna()]
        if not wins.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            for level, group in wins.groupby(level_column):
                group = group.sort_values("training_games")
                ax3.plot(
                    group["training_games"],
                    group["mean_plies_to_win"],
                    marker="o",
                    label=f"SF {level}",
                )
            ax3.set_xlabel("Training games")
            ax3.set_ylabel("Mean plies to win")
            ax3.set_title("Checkpoint plies to win")
            ax3.legend()
            ax3.grid(alpha=0.3)
            _save_or_show(fig3, output_dir, "checkpoint_plies_to_win.png", show=show)


def analyse_results(
    log_file=None,
    log_pattern="logs/games_worker_*.jsonl",
    worker_id=None,
    bandit_type=None,
    output_dir=None,
    show=True,
    games_per_segment=5,
    window_games=5,
    checkpoint_csv="auto",
):
    df = load_training_logs(
        log_file=log_file,
        log_pattern=log_pattern,
        worker_id=worker_id,
        bandit_type=bandit_type,
    )
    if df.empty:
        print("No data found in the selected log file(s).")
        return

    df = _prepare_logs(df)
    output_dir = _resolve_output_dir(output_dir)

    _print_tables(df, games_per_segment=games_per_segment, window_games=window_games)
    _plot_outcomes(df, output_dir, show)
    _plot_rewards(df, output_dir, show, window_games)
    _plot_arm_usage(df, output_dir, show)
    _plot_exploration_vs_exploitation(df, output_dir, show, window_games)

    if checkpoint_csv == "auto":
        checkpoint_csv = _find_checkpoint_csv(worker_id, bandit_type)
    if checkpoint_csv:
        _plot_checkpoint_evaluation(checkpoint_csv, output_dir, show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        help="Analyze a single log file instead of all worker logs.",
    )
    parser.add_argument(
        "--log-pattern",
        default="logs/games_worker_*.jsonl",
        help="Glob pattern used when --log-file is not provided.",
    )
    parser.add_argument("--worker-id", help="Analyze one worker log by id.")
    parser.add_argument(
        "--bandit-type",
        choices=["basic_linucb", "neural_linucb"],
        help="Filter logs by bandit type.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where PNG plots should be saved. Relative paths are under project/.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open Matplotlib windows; useful when saving plots.",
    )
    parser.add_argument(
        "--games-per-segment",
        type=int,
        default=5,
        help="Segment size for printed training summaries.",
    )
    parser.add_argument(
        "--window-games",
        type=int,
        default=5,
        help="Rolling window, in games, for exploration/exploitation proxy curves.",
    )
    parser.add_argument(
        "--checkpoint-csv",
        default="auto",
        help="Checkpoint evaluation CSV to plot, 'auto' to infer from worker id, or empty to skip.",
    )
    args = parser.parse_args()

    checkpoint_csv = args.checkpoint_csv
    if checkpoint_csv == "":
        checkpoint_csv = None

    analyse_results(
        log_file=args.log_file,
        log_pattern=args.log_pattern,
        worker_id=args.worker_id,
        bandit_type=args.bandit_type,
        output_dir=args.output_dir,
        show=not args.no_show,
        games_per_segment=args.games_per_segment,
        window_games=args.window_games,
        checkpoint_csv=checkpoint_csv,
    )
