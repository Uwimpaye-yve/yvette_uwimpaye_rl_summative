import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment.custom_env import CHWEnv
from training.utils import evaluate

EPISODES = 1500  # same training budget as DQN


def make_net(out_size):
    return nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out_size))


def discounted_returns(rewards, gamma):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def gae(rewards, values, gamma, lam):
    """Generalized Advantage Estimation for one episode."""
    values = list(values) + [0.0]
    out, a = [], 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        a = delta + gamma * lam * a
        out.append(a)
    return out[::-1]


def collect(env, actor, n_episodes, seed_base):
    """Play n episodes with the current policy. Returns a list of (observations, actions, rewards)."""
    batch = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + k)
        done, O, A, R = False, [], [], []
        while not done:
            with torch.no_grad():
                action = int(Categorical(logits=actor(torch.tensor(obs))).sample())
            O.append(obs)
            A.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            R.append(reward)
            done = terminated or truncated
        batch.append((O, A, R))
    return batch


def train_pg(algo, name, lr=1e-3, gamma=0.99, batch_episodes=5, ent_coef=0.01,
             lam=0.95, vf_coef=0.5, clip=0.2, epochs=4, baseline=True, seed=0):
    torch.manual_seed(seed)
    env = CHWEnv()
    actor, critic = make_net(11), make_net(1)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    log, ep = [], 0
    while ep < EPISODES:
        batch = collect(env, actor, batch_episodes, seed * 10000 + ep)
        obs = torch.tensor(np.array([o for O, _, _ in batch for o in O]))
        acts = torch.tensor([a for _, A, _ in batch for a in A])
        returns = torch.tensor([g for _, _, R in batch for g in discounted_returns(R, gamma)], dtype=torch.float32)

        if algo == "reinforce":
            adv = (returns - returns.mean()) / (returns.std() + 1e-8) if baseline else returns
            dist = Categorical(logits=actor(obs))
            loss = -(dist.log_prob(acts) * adv).mean() - ent_coef * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:  # "a2c" or "ppo": use the critic to compute advantages
            with torch.no_grad():
                advs = []
                for O, _, R in batch:
                    values = critic(torch.tensor(np.array(O))).squeeze(-1).tolist()
                    advs += gae(R, values, gamma, lam)
                adv = torch.tensor(advs, dtype=torch.float32)
                value_targets = adv + critic(obs).squeeze(-1)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                old_logp = Categorical(logits=actor(obs)).log_prob(acts)

            for _ in range(epochs if algo == "ppo" else 1):
                dist = Categorical(logits=actor(obs))
                logp = dist.log_prob(acts)
                if algo == "ppo":
                    ratio = torch.exp(logp - old_logp)
                    policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
                else:
                    policy_loss = -(logp * adv).mean()
                value_loss = nn.functional.mse_loss(critic(obs).squeeze(-1), value_targets)
                loss = policy_loss + vf_coef * value_loss - ent_coef * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        entropy = dist.entropy().mean().item()
        for _, _, R in batch:
            log.append({"episode": ep, "reward": sum(R), "entropy": entropy, "loss": loss.item()})
            if ep % 100 == 0:
                print(f"[{algo} {name}] episode {ep}  reward {sum(R):.1f}  entropy {entropy:.2f}")
            ep += 1

    torch.save(actor.state_dict(), f"models/pg/{algo}_{name}.pt")
    pd.DataFrame(log).to_csv(f"logs/{algo}_{name}.csv", index=False)
    # policy-gradient agents learn a probability per action, so we test them by sampling from it
    eval_mean, eval_std = evaluate(lambda o: int(Categorical(logits=actor(torch.tensor(o))).sample()))
    train_last100 = pd.DataFrame(log)["reward"].tail(100).mean()
    return train_last100, eval_mean, eval_std


DEFAULTS = {
    "reinforce": dict(lr=3e-3, gamma=0.99, batch_episodes=5, ent_coef=0.01, baseline=True),
    "a2c": dict(lr=3e-3, gamma=0.99, batch_episodes=1, ent_coef=0.01, lam=0.95, vf_coef=0.5),
    "ppo": dict(lr=1e-3, gamma=0.99, batch_episodes=5, ent_coef=0.01, lam=0.95, vf_coef=0.5, clip=0.2, epochs=10),
}

RUNS = {
    "reinforce": [
        dict(name="run1_baseline"),
        dict(name="run2_lr_low", lr=1e-3),
        dict(name="run3_lr_high", lr=1e-2),
        dict(name="run4_gamma_090", gamma=0.90),
        dict(name="run5_gamma_095", gamma=0.95),
        dict(name="run6_batch_1", batch_episodes=1),
        dict(name="run7_batch_20", batch_episodes=20),
        dict(name="run8_entropy_0", ent_coef=0.0),
        dict(name="run9_entropy_005", ent_coef=0.05),
        dict(name="run10_no_baseline", baseline=False),
    ],
    "a2c": [
        dict(name="run1_baseline"),
        dict(name="run2_lr_low", lr=1e-3),
        dict(name="run3_lr_high", lr=1e-2),
        dict(name="run4_gamma_090", gamma=0.90),
        dict(name="run5_lambda_100", lam=1.0),
        dict(name="run6_lambda_050", lam=0.5),
        dict(name="run7_batch_5", batch_episodes=5),
        dict(name="run8_entropy_0", ent_coef=0.0),
        dict(name="run9_entropy_005", ent_coef=0.05),
        dict(name="run10_vf_100", vf_coef=1.0),
    ],
    "ppo": [
        dict(name="run1_baseline"),
        dict(name="run2_lr_low", lr=3e-4),
        dict(name="run3_lr_high", lr=3e-3),
        dict(name="run4_gamma_090", gamma=0.90),
        dict(name="run5_clip_010", clip=0.1),
        dict(name="run6_clip_030", clip=0.3),
        dict(name="run7_epochs_4", epochs=4),
        dict(name="run8_batch_10", batch_episodes=10),
        dict(name="run9_entropy_005", ent_coef=0.05),
        dict(name="run10_lambda_100", lam=1.0),
    ],
}

if __name__ == "__main__":
    # usage: python -m training.pg_training reinforce|a2c|ppo|all [--quick]
    algos = ["reinforce", "a2c", "ppo"] if sys.argv[1] == "all" else [sys.argv[1]]
    if "--quick" in sys.argv:
        EPISODES = 30

    for algo in algos:
        runs = RUNS[algo][:1] if "--quick" in sys.argv else RUNS[algo]
        results = []
        for run in runs:
            settings = {**DEFAULTS[algo], **run}
            train_last100, eval_mean, eval_std = train_pg(algo, **settings)
            results.append({**settings, "train_mean_last100": round(train_last100, 2),
                            "eval_mean": round(eval_mean, 2), "eval_std": round(eval_std, 2)})

        table = pd.DataFrame(results)
        table.to_csv(f"logs/{algo}_results.csv", index=False)
        print(table.to_string(index=False))

        best = table.loc[table["eval_mean"].idxmax(), "name"]
        shutil.copy(f"models/pg/{algo}_{best}.pt", f"models/pg/best_{algo}.pt")
        print(f"Best {algo} run:", best)