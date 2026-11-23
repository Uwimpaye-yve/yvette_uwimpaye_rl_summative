import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.utils import evaluate

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
WINDOW = 50  # moving-average window (episodes)

# load the training log of the best run of each algorithm
best_logs, best_rows = {}, {}
for algo in ALGOS:
    results = pd.read_csv(f"logs/{algo}_results.csv")
    row = results.loc[results["eval_mean"].idxmax()]
    best_rows[algo] = row
    best_logs[algo] = pd.read_csv(f"logs/{algo}_{row['name']}.csv")
    best_logs[algo]["ma"] = best_logs[algo]["reward"].rolling(WINDOW).mean()

# 1. cumulative reward per episode for each best model
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for ax, algo in zip(axes.flat, ALGOS):
    log = best_logs[algo]
    ax.plot(log["episode"], log["reward"], alpha=0.3, label="episode reward")
    ax.plot(log["episode"], log["ma"], label=f"{WINDOW}-episode average")
    ax.set_title(f"{NAMES[algo]} ({best_rows[algo]['name']})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend()
fig.suptitle("Cumulative reward per episode - best model of each method")
fig.tight_layout()
fig.savefig("assets/1_cumulative_rewards.png", dpi=150)

# 2. training stability: DQN loss and policy-gradient entropy
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
log = best_logs["dqn"]
axes[0, 0].plot(log["episode"], log["loss"].rolling(20).mean())
axes[0, 0].set_title("DQN - TD loss (MSE, 20-episode average)")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Loss")
for ax, algo in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["reinforce", "a2c", "ppo"]):
    ax.plot(best_logs[algo]["episode"], best_logs[algo]["entropy"])
    ax.set_title(f"{NAMES[algo]} - policy entropy")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy (nats)")
fig.suptitle("Training stability")
fig.tight_layout()
fig.savefig("assets/2_training_stability.png", dpi=150)

# 3. convergence: all methods on one plot + episodes to converge
plt.figure(figsize=(10, 5))
print("\nEpisodes to converge (first episode where the 50-episode average")
print("gets within 10% of its final value and the final value is the mean of the last 100 episodes):")
for algo in ALGOS:
    log = best_logs[algo]
    plt.plot(log["episode"], log["ma"], label=NAMES[algo])
    final = log["reward"].tail(100).mean()
    threshold = final - 0.1 * abs(final)
    reached = log.index[log["ma"] >= threshold]
    episode = int(log.loc[reached[0], "episode"]) if len(reached) else None
    print(f"  {NAMES[algo]:10s} final avg = {final:6.2f}   converged at episode {episode}")
plt.xlabel("Episode")
plt.ylabel(f"Reward ({WINDOW}-episode average)")
plt.title("Convergence of the best model of each method")
plt.legend()
plt.tight_layout()
plt.savefig("assets/3_convergence.png", dpi=150)

# 4. generalization: evaluation on 50 unseen starting caseloads
rng = np.random.default_rng(0)
random_mean, random_std = evaluate(lambda o: int(rng.integers(11)))
labels = ["Random"] + [NAMES[a] for a in ALGOS]
means = [random_mean] + [best_rows[a]["eval_mean"] for a in ALGOS]
stds = [random_std] + [best_rows[a]["eval_std"] for a in ALGOS]
plt.figure(figsize=(8, 5))
plt.bar(labels, means, yerr=stds, capsize=6, color=["grey", "C0", "C1", "C2", "C3"])
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Mean total reward (50 unseen episodes)")
plt.title("Generalization to unseen initial states")
plt.tight_layout()
plt.savefig("assets/4_generalization.png", dpi=150)

print("\nGeneralization (50 unseen starting caseloads):")
for label, m, s in zip(labels, means, stds):
    print(f"  {label:10s} {m:6.2f} +/- {s:.2f}")
print("\nPlots saved in the assets folder.")