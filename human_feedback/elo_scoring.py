#!/usr/bin/env python3
import os
import sys
import glob
import json
import argparse
import pandas as pd
from omegaconf import OmegaConf

# Allow project imports
sys.path.append(os.environ.get("ROOT_PATH", ""))


# ---------------------------------------------------------
# Config loader
# ---------------------------------------------------------
def load_cfg():
    root = os.environ["ROOT_PATH"]
    cfg_file = os.path.join(root, "cfg", "generate.yaml")
    return OmegaConf.load(cfg_file)


# ---------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------
def update_elo(r1, r2, result):
    K = 32
    exp1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
    exp2 = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / 400.0))
    new1 = r1 + K * (result - exp1)
    new2 = r2 + K * (1.0 - result - exp2)
    return new1, new2


def elo_scores(df: pd.DataFrame):
    videos = pd.concat([df["Video 1"], df["Video 2"]]).unique()
    ratings = {v: 1500.0 for v in videos}

    for _, row in df.iterrows():
        v1, v2, sel = row["Video 1"], row["Video 2"], row["Selected"]
        if sel == 1.0:
            res = 1.0
        elif sel == 2.0:
            res = 0.0
        else:
            res = 0.5

        ratings[v1], ratings[v2] = update_elo(ratings[v1], ratings[v2], res)

    max_r = max(ratings.values())
    min_r = min(ratings.values())
    if max_r == min_r:
        return {k: 0.5 for k in ratings.keys()}

    return {k: (v - min_r) / (max_r - min_r) for k, v in ratings.items()}


# ---------------------------------------------------------
# Feedback grouping
# ---------------------------------------------------------
def group_feedback(df: pd.DataFrame) -> pd.DataFrame:
    def split(x):
        if isinstance(x, str) and x:
            return x.split(", ")
        return []

    vids = set(df["Video 1"]).union(df["Video 2"])
    fb = {v: {"Positive Feedback": [], "Negative Feedback": []} for v in vids}

    for _, row in df.iterrows():
        v1, p1, n1 = row["Video 1"], row["Positive Feedback 1"], row["Negative Feedback 1"]
        v2, p2, n2 = row["Video 2"], row["Positive Feedback 2"], row["Negative Feedback 2"]

        fb[v1]["Positive Feedback"].extend(split(p1))
        fb[v1]["Negative Feedback"].extend(split(n1))
        fb[v2]["Positive Feedback"].extend(split(p2))
        fb[v2]["Negative Feedback"].extend(split(n2))

    for v in fb.values():
        v["Positive Feedback"] = sorted(set(v["Positive Feedback"]))
        v["Negative Feedback"] = sorted(set(v["Negative Feedback"]))

    df_fb = pd.DataFrame.from_dict(fb, orient="index")
    df_fb.index.name = "Video"
    return df_fb


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_id", required=True, type=int, help="generation id (upper bound)")
    args = parser.parse_args()
    GEN_ID = args.gen_id

    cfg = load_cfg()
    root = os.environ["ROOT_PATH"]
    baseline = cfg.evolution.baseline
    run_id = cfg.data_paths.run

    # -----------------------------------------------------
    # Load responses from ALL generations 0..GEN_ID
    # -----------------------------------------------------
    cols = [
        "Video 1", "Video 2", "Selected",
        "Positive Feedback 1", "Negative Feedback 1",
        "Positive Feedback 2", "Negative Feedback 2",
    ]
    df = pd.DataFrame(columns=cols)

    for gid in range(GEN_ID + 1):
        hf_dir = os.path.join(root, "human_feedback", f"generation_{gid}")
        response_paths = glob.glob(os.path.join(hf_dir, "responses_*.csv"))
        for rp in response_paths:
            df_r = pd.read_csv(rp)
            df = pd.concat([df, df_r], ignore_index=True)

    if df.empty:
        print("No survey responses found for 0..GEN_ID.")
        return

    # -----------------------------------------------------
    # Elo + Feedback
    # -----------------------------------------------------
    scores = elo_scores(df)
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    scores_df = pd.DataFrame(scores_sorted, columns=["Video", "Score"])

    fb_df = group_feedback(df)
    combined = pd.merge(scores_df, fb_df, on="Video")

    # -----------------------------------------------------
    # Write fitness for ALL videos in CSVs (ALL generations)
    # -----------------------------------------------------
    db_root = os.path.join(root, "database", baseline, str(run_id))

    for _, row in combined.iterrows():
        video_name = row["Video"]
        hf_score = float(row["Score"])
        pos_fb = ", ".join(row["Positive Feedback"])
        neg_fb = ", ".join(row["Negative Feedback"])

        base = os.path.basename(video_name).replace(".mp4", "")
        parts = base.split("_")
        if len(parts) < 3:
            continue

        gen_str, ctr_str = parts[-2], parts[-1]
        try:
            gen = int(gen_str)
            counter = int(ctr_str)
        except ValueError:
            continue

        # locate island
        pattern = os.path.join(
            db_root,
            "island_*",
            "generated_fns",
            f"{gen}_{counter}.txt"
        )
        matches = glob.glob(pattern)
        if not matches or len(matches) != 1:
            print(f"Warning: could not locate individual for video {video_name}")
            continue

        island_dir = os.path.dirname(os.path.dirname(matches[0]))

        # Write fitness file
        fitness_dir = os.path.join(island_dir, "fitness_scores")
        os.makedirs(fitness_dir, exist_ok=True)
        fitness_file = os.path.join(fitness_dir, f"{gen}_{counter}.txt")
        with open(fitness_file, "w") as f:
            json.dump({"fitness": hf_score}, f)

        # Write feedback text
        hf_text_dir = os.path.join(island_dir, "human_feedback")
        os.makedirs(hf_text_dir, exist_ok=True)
        hf_text_file = os.path.join(hf_text_dir, f"{gen}_{counter}.txt")
        with open(hf_text_file, "w") as f:
            f.write("Positive:\n")
            f.write(pos_fb + "\n\n")
            f.write("Negative:\n")
            f.write(neg_fb + "\n")

    print("Elo-based fitness written for ALL individuals appearing in CSVs (gens 0..GEN_ID).")


if __name__ == "__main__":
    main()
