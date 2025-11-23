#!/usr/bin/env python3
import os
import time
import glob
import argparse
import numpy as np

from omegaconf import OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_agent.HumanoidEnv import HumanoidEnv


# ---------------------------------------------------------
# Load YAML config WITHOUT Hydra
# ---------------------------------------------------------
def load_cfg():
    root = os.environ["ROOT_PATH"]
    cfg_file = os.path.join(root, "cfg", "generate.yaml")
    cfg = OmegaConf.load(cfg_file)
    return cfg


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def load_velocity_log(path):
    vals = []
    with open(path, "r") as f:
        for line in f:
            try:
                vals.append(float(line.strip()))
            except:
                pass
    return np.array(vals) if vals else None


def round_to_checkpoint(idx, interval=1000):
    return int(round(idx / interval) * interval)


def run_eps(model, env, n=3):
    for _ in range(n):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(action)
            env.render("human")


# ---------------------------------------------------------
# Main logic
# ---------------------------------------------------------
def visualize(generation_id):
    cfg = load_cfg()

    root = os.environ["ROOT_PATH"]
    baseline = cfg.evolution.baseline
    run_id = cfg.data_paths.run

    db_root = os.path.join(root, "database", baseline, str(run_id))
    print("DB PATH:", db_root)

    island_dirs = sorted(glob.glob(os.path.join(db_root, "island_*")))
    if not island_dirs:
        print("No islands found")
        return

    for island_dir in island_dirs:
        island_id = int(island_dir.split("_")[-1])

        print(f"\n=== ISLAND {island_id} | GEN {generation_id} ===")

        vel_dir = os.path.join(island_dir, "velocity_logs")
        model_dir = os.path.join(island_dir, "model_checkpoints")
        vec_dir = os.path.join(island_dir, "vecnormalize")

        pattern = os.path.join(vel_dir, f"velocity_{generation_id}_*.txt")
        individuals = sorted(glob.glob(pattern))
        if not individuals:
            print("No individuals found for this generation")
            continue

        for vel_file in individuals:
            _, gen, counter = os.path.basename(vel_file).replace(".txt", "").split("_")
            gen = int(gen)
            counter = int(counter)

            vel = load_velocity_log(vel_file)
            if vel is None or len(vel) == 0:
                continue

            best_step = round_to_checkpoint(int(np.argmax(vel)), 10000)

            model_path = os.path.join(model_dir, f"SAC_{gen}_{counter}_{best_step}.zip")
            vec_path = os.path.join(vec_dir, f"{gen}_{counter}_{best_step}.pkl")

            if not os.path.exists(model_path) or not os.path.exists(vec_path):
                continue

            print(f"INDIVIDUAL {gen}_{counter}")
            print("CKPT:", model_path)
            print("VEC:", vec_path)

            reward_fn_file = os.path.join(
                island_dir, "generated_fns", f"{gen}_{counter}.txt"
            )

            with open(reward_fn_file, "r") as f:
                reward_func_str = f.read()

            # ----------- DUMMY SAFE PATHS (never written) -----------
            reward_history_file = "/tmp/dummy_reward.json"
            model_checkpoint_file = "/tmp/dummy_ckpt.h5"
            velocity_file = "/tmp/dummy_vel.txt"
            # ---------------------------------------------------------

            def make_env():
                return HumanoidEnv(
                    reward_func_str=reward_func_str,
                    counter=counter,
                    generation_id=gen,
                    island_id=island_id,
                    reward_history_file=reward_history_file,
                    model_checkpoint_file=model_checkpoint_file,
                    velocity_file=velocity_file,
                    render_mode="human"
                )

            env = DummyVecEnv([make_env])
            env = VecNormalize.load(vec_path, env)
            env.training = False
            env.norm_reward = False

            model = SAC.load(model_path, env=env)

            run_eps(model, env, n=5)
            env.close()

            # Safety: also close low-level viewer if needed
            try:
                env.envs[0].close()
            except:
                pass

            time.sleep(5)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    args = parser.parse_args()

    visualize(args.generation)
# ---------------------------------------------------------