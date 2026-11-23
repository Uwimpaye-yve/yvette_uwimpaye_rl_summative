from environment.custom_env import CHWEnv

env = CHWEnv(render_mode="human")
obs, _ = env.reset(seed=1)
env.render()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Week {env.week}: action={action}, reward={reward:.1f}")
    done = terminated or truncated

env.renderer.save("assets/environment.png")
print("Screenshot saved to assets/environment.png")
env.close()