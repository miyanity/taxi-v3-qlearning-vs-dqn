import gymnasium as gym

env = gym.make("Taxi-v3")
s, info = env.reset()
print("state:", s)
print("actions:", env.action_space.n)
env.close()