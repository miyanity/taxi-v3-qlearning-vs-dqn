import os

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# simple neural network that predicts q values
class qnet(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.net(x)


# turn the taxi state number into a one hot vector
def one_hot(state, num_states):
    x = np.zeros(num_states, dtype=np.float32)
    x[state] = 1.0
    return x


def main():
    # make the environment
    env = gym.make("Taxi-v3")
    os.makedirs("results", exist_ok=True)
    
    # get how many states and actions there are
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # make the main network and the target network
    policy = qnet(num_states, num_actions)
    target = qnet(num_states, num_actions)
    target.load_state_dict(policy.state_dict())
    target.eval()

    # optimizer
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # replay buffer stores experiences
    replay = deque(maxlen=50000)

    # settings
    episodes = 5000
    gamma = 0.99
    batch_size = 64
    min_buffer = 2000

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    target_update_steps = 1000
    steps = 0

    rewards_per_episode = []

    # training loop
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            steps += 1

            # pick random action sometimes (explore)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # pick best action from the network
                with torch.no_grad():
                    s_vec = torch.tensor(one_hot(state, num_states)).unsqueeze(0)
                    q_values = policy(s_vec)
                    action = int(torch.argmax(q_values, dim=1).item())

            # take the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # save the experience
            replay.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            # only train if we have enough experiences saved
            if len(replay) >= min_buffer:
                batch = random.sample(replay, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # turn states into one hot vectors
                s = torch.tensor([one_hot(x, num_states) for x in states])
                ns = torch.tensor([one_hot(x, num_states) for x in next_states])

                a = torch.tensor(actions).long().unsqueeze(1)
                r = torch.tensor(rewards).float().unsqueeze(1)
                d = torch.tensor(dones).float().unsqueeze(1)

                # current q(s,a)
                q_sa = policy(s).gather(1, a)

                # target: r + gamma * max(q_target(next_state))
                with torch.no_grad():
                    max_q_next = target(ns).max(dim=1, keepdim=True)[0]
                    y = r + gamma * (1.0 - d) * max_q_next

                # train step
                loss = loss_fn(q_sa, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update target network sometimes
                if steps % target_update_steps == 0:
                    target.load_state_dict(policy.state_dict())

        # slowly explore less over time
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # save reward for this episode
        rewards_per_episode.append(total_reward)

        # little progress print so you know its working
        if (episode + 1) % 500 == 0:
            avg = np.mean(rewards_per_episode[-100:])
            print(f"episode {episode+1} | avg last 100 rewards: {avg:.2f} | epsilon: {epsilon:.2f}")

    # save rewards so we can plot later
    np.savetxt("results/dqn_rewards.csv", rewards_per_episode, delimiter=",")

    env.close()
    print("dqn training complete!")


if __name__ == "__main__":
    main()