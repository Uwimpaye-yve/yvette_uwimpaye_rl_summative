import random
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

from environment.custom_env import CHWEnv
from training.utils import evaluate

EPISODES = 1500  # same training budget for every algorithm


def make_net():
    return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 11))


def train_dqn(name, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64,
              eps_decay=0.995, target_update=200, eps_min=0.05, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    env = CHWEnv()
    q_net, target_net = make_net(), make_net()
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(storage=LazyTensorStorage(buffer_size), batch_size=batch_size)  # TorchRL

    eps, steps, log = 1.0, 0, []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        done, total, losses = False, 0, []
        while not done:
            # epsilon-greedy action
            if random.random() < eps:
                action = int(env.action_space.sample())
            else:
                with torch.no_grad():
                    action = int(q_net(torch.tensor(obs)).argmax())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(TensorDict({
                "obs": torch.tensor(obs),
                "action": torch.tensor(action),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_obs": torch.tensor(next_obs),
                "done": torch.tensor(float(terminated)),
            }, batch_size=[]))
            obs, total, steps = next_obs, total + reward, steps + 1

            # learn from a random batch of past experience
            if len(buffer) >= batch_size:
                b = buffer.sample()
                q_values = q_net(b["obs"]).gather(1, b["action"].long().unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target = b["reward"] + gamma * target_net(b["next_obs"]).max(1).values * (1 - b["done"])
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if steps % target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

        eps = max(eps_min, eps * eps_decay)
        log.append({"episode": ep, "reward": total, "loss": np.mean(losses) if losses else 0, "epsilon": eps})

        if ep % 100 == 0:
            print(f"[{name}] episode {ep}  reward {total:.1f}  epsilon {eps:.2f}")

    torch.save(q_net.state_dict(), f"models/dqn/{name}.pt")
    pd.DataFrame(log).to_csv(f"logs/dqn_{name}.csv", index=False)
    eval_mean, eval_std = evaluate(lambda o: int(q_net(torch.tensor(o)).argmax()))
    train_last100 = pd.DataFrame(log)["reward"].tail(100).mean()
    return train_last100, eval_mean, eval_std


RUNS = [
    dict(name="run1_baseline"),
    dict(name="run2_lr_low", lr=1e-4),
    dict(name="run3_lr_high", lr=5e-3),
    dict(name="run4_gamma_090", gamma=0.90),
    dict(name="run5_buffer_small", buffer_size=1000),
    dict(name="run6_batch_128", batch_size=128),
    dict(name="run7_eps_fast", eps_decay=0.99),
    dict(name="run8_eps_slow", eps_decay=0.998),
    dict(name="run9_target_50", target_update=50),
    dict(name="run10_target_1000", target_update=1000),
]

if __name__ == "__main__":
    if "--quick" in sys.argv:  # fast check that everything works
        EPISODES = 30
        RUNS = RUNS[:1]

    defaults = dict(lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64, eps_decay=0.995, target_update=200)
    results = []
    for run in RUNS:
        settings = {**defaults, **run}
        train_last100, eval_mean, eval_std = train_dqn(**settings)
        results.append({**settings, "train_mean_last100": round(train_last100, 2),
                        "eval_mean": round(eval_mean, 2), "eval_std": round(eval_std, 2)})

    table = pd.DataFrame(results)
    table.to_csv("logs/dqn_results.csv", index=False)
    print(table.to_string(index=False))

    best = table.loc[table["eval_mean"].idxmax(), "name"]
    shutil.copy(f"models/dqn/{best}.pt", "models/dqn/best_dqn.pt")
    print("Best DQN run:", best)