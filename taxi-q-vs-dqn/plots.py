import os
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, window=100):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

def main():
    # make sure results folder exists
    os.makedirs("results", exist_ok=True)

    # load reward files
    q_rewards = np.loadtxt("results/qlearning_rewards.csv", delimiter=",")
    d_rewards = np.loadtxt("results/dqn_rewards.csv", delimiter=",")

    # smooth curves
    q_ma = moving_average(q_rewards)
    d_ma = moving_average(d_rewards)

    # create plot
    plt.figure(figsize=(8, 5))
    plt.plot(q_rewards, alpha=0.3, label="q-learning raw")
    plt.plot(d_rewards, alpha=0.3, label="dqn raw")
    plt.plot(q_ma, label="q-learning avg")
    plt.plot(d_ma, label="dqn avg")

    plt.title("taxi-v3: q-learning vs dqn")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.legend()

    # save and show
    plt.savefig("results/learning_curves.png", dpi=200)
    plt.show()

    print("plot saved to results/learning_curves.png")

if __name__ == "__main__":
    main()