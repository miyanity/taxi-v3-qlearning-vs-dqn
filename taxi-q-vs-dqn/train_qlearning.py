import gymnasium as gym
import numpy as np

# make the taxi environment
env = gym.make("Taxi-v3")

# get how many states and actions there are
num_states = env.observation_space.n
num_actions = env.action_space.n

# make the q table and fill it with zeros
# rows = states, columns = actions
Q = np.zeros((num_states, num_actions))

# settings for learning
alpha = 0.1        # how fast the agent learns
gamma = 0.99       # how much future rewards matter
epsilon = 1.0      # start by exploring a lot
epsilon_decay = 0.995
epsilon_min = 0.05
episodes = 10000

# list to keep track of rewards each episode
rewards_per_episode = []

# loop through all episodes
for episode in range(episodes):
    # reset the environment at the start
    state, _ = env.reset()
    done = False
    total_reward = 0

    # keep going until the episode ends
    while not done:
        # decide whether to explore or use what we know
        if np.random.rand() < epsilon:
            # pick a random action
            action = env.action_space.sample()
        else:
            # pick the best action from the q table
            action = np.argmax(Q[state])

        # take the action and see what happens
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # update the q value using the q learning formula
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        # move to the next state
        state = next_state
        total_reward += reward

    # slowly explore less over time
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # save the total reward for this episode
    rewards_per_episode.append(total_reward)

# save the rewards so we can plot them later
np.savetxt("results/qlearning_rewards.csv", rewards_per_episode, delimiter=",")

# close the environment
env.close()

print("q learning training complete!")