import numpy as np
from environment.custom_env import CHWEnv

env = CHWEnv()
obs, _ = env.reset(seed=0)
print("Observation shape:", obs.shape)

total, done = 0, False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated
print("Random agent total reward:", round(total, 2))