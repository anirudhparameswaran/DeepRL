# import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
# import multiarmed bandit class and algorithms
from functions_01_1 import MultiArmedBandit, epsilon_greedy, ucb

# fix a random seed
np.random.seed(41)

# number of bandits
K = 10
# means
mus = np.random.normal(0, 1, K)
# instantiation of multiarmed bandit
multiArmedBandit = MultiArmedBandit(mus)

# set number of actions
n = 10000

# 
avg_reward_eps, opt_eps, reg_eps = epsilon_greedy(multiArmedBandit , n, epsilon=0.1)
avg_reward_eps_tv, opt_eps_tv, reg_eps_tv = epsilon_greedy(multiArmedBandit, n, epsilon = 0.1, time_varying=True, delta = 1)
avg_reward_ucb, opt_ucb, reg_ucb = ucb(multiArmedBandit , n, c = 2)

# a) plot reward

plt.figure(figsize=(10, 5))
plt.plot(avg_reward_eps, label="ε=0.1")
plt.plot(avg_reward_ucb, label="UCB")
plt.plot(avg_reward_eps_tv, label="ε_t=1/t")

plt.title("Reward")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.legend()
plt.grid()
plt.show()

# b) plot percentage of optimal actions

plt.figure(figsize=(10, 5))
plt.plot(opt_eps, label="ε=0.1")
plt.plot(opt_ucb, label="UCB")
plt.plot(opt_eps_tv, label="ε_t=1/t")

plt.title("Optimal Action (%)")
plt.xlabel("Steps")
plt.ylabel("Proportion")
plt.legend()
plt.grid()
plt.show()

# c) plot regret

plt.figure(figsize=(10, 5))
plt.plot(reg_eps, label="ε=0.1")
plt.plot(reg_ucb, label="UCB")
plt.plot(reg_eps_tv, label="ε_t=1/t")

plt.title("Cumulative Regret")
plt.xlabel("Steps")
plt.ylabel("Regret")
plt.legend()
plt.grid()
plt.show()