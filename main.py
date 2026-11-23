import sys

import pandas as pd
import torch
from torch.distributions import Categorical

from environment.custom_env import CHWEnv
from environment.rendering import action_name
from training.dqn_training import make_net as make_dqn_net
from training.pg_training import make_net as make_pg_net

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
EPISODES = 3


def pick_algo():
    """Use the algorithm given on the command line, otherwise the best one overall."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    scores = {a: pd.read_csv(f"logs/{a}_results.csv")["eval_mean"].max() for a in ALGOS}
    return max(scores, key=scores.get)


algo = pick_algo()
if algo == "dqn":
    net = make_dqn_net()
    net.load_state_dict(torch.load("models/dqn/best_dqn.pt"))
    act = lambda o: int(net(torch.tensor(o)).argmax())
else:
    net = make_pg_net(11)
    net.load_state_dict(torch.load(f"models/pg/best_{algo}.pt"))
    act = lambda o: int(Categorical(logits=net(torch.tensor(o))).sample())

print(f"Running best {algo.upper()} agent for {EPISODES} episodes (unseen starting caseloads)")
env = CHWEnv(render_mode="human")
for ep in range(EPISODES):
    obs, _ = env.reset(seed=500000 + ep)  # seeds never used in training
    env.render()
    print(f"\n=== Episode {ep + 1}  starting risk levels (hidden from agent): {env.risk.tolist()} ===")
    done, total = False, 0
    while not done:
        with torch.no_grad():
            action = act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
        print(f"Week {env.week:2d} | {action_name(action):36s} | reward {reward:5.1f} | "
              f"total {total:6.1f} | youth in crisis: {info['crisis']}")
    outcome = "caseload out of control (3 youth in crisis)" if terminated else "completed all 20 weeks"
    print(f"Episode {ep + 1} finished: {outcome}, total reward {total:.1f}")
env.close()