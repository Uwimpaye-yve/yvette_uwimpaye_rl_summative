import numpy as np
import torch
from environment.custom_env import CHWEnv


def evaluate(act, episodes=50, seed_start=100000):
    """Test an agent on 50 unseen starting caseloads (seeds never used in training)."""
    env = CHWEnv()
    totals = []
    with torch.no_grad():
        for k in range(episodes):
            obs, _ = env.reset(seed=seed_start + k)
            done, total = False, 0
            while not done:
                obs, r, terminated, truncated, _ = env.step(act(obs))
                total += r
                done = terminated or truncated
            totals.append(total)
    return float(np.mean(totals)), float(np.std(totals))