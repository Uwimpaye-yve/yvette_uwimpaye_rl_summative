# CHW Triage & Referral Resource Allocator (Reinforcement Learning)

A reinforcement learning environment based on my capstone: a community health worker (CHW)
in Kigali supports 5 youth over 20 weeks. Each youth's emotional-distress level is hidden and
can get worse if they are not seen. Each week the CHW chooses one action: screen a youth,
refer a youth to the health centre (only 2 treatment slots), or run a group session at school.
The agent must learn whom to screen, when to refer, and how to use limited referral slots.

Four algorithms are compared on the same environment: DQN, REINFORCE, A2C and PPO (PyTorch + TorchRL).

## Run the project

```bash
git clone https://github.com/Uwimpaye-yve/yvette_uwimpaye_rl_summative.git
cd yvette_uwimpaye_rl_summative
uv sync
uv run main.py            # plays the best trained agent (GUI + terminal output)
uv run main.py dqn        # or choose: dqn, reinforce, a2c, ppo
```

## Retrain and reproduce the results

```bash
uv run python -m training.dqn_training          # 10 DQN runs
uv run python -m training.pg_training reinforce # 10 REINFORCE runs
uv run python -m training.pg_training a2c       # 10 A2C runs
uv run python -m training.pg_training ppo       # 10 PPO runs
uv run python plots.py                          # figures in assets/
```

## Project structure

```
environment/custom_env.py   Gymnasium environment (actions, observations, rewards)
environment/rendering.py    Pygame visualisation
training/dqn_training.py    DQN with TorchRL replay buffer + 10 hyperparameter runs
training/pg_training.py     REINFORCE, A2C, PPO + 10 hyperparameter runs each
training/utils.py           Evaluation on 50 unseen starting caseloads
models/                     Saved models (best_*.pt = best run of each method)
logs/                       Training logs and results tables (CSV)
assets/                     Environment screenshot and result plots
main.py                     Runs the best agent with visualisation
```

## Results (mean reward on 50 unseen caseloads)

| Method | Best run | Reward |
|---|---|---|
| A2C | lambda = 0.5 | 13.41 ± 6.74 |
| DQN | gamma = 0.90 | 12.54 ± 7.23 |
| PPO | gamma = 0.90 | 5.62 ± 11.61 |
| REINFORCE | gamma = 0.90 | 1.56 ± 10.90 |
| Random | – | -10.31 ± 10.90 |