from gridworld import GridWorld
from functions_01_3 import solve_Bellman_expectation, plot_V
from functions_01_5 import SARSA, Q_learning
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

# initialize size, battery position, obstacle positions, epsilon
size_ = 6
battery_pos_ = (4,3)
obstacles_ = {(0, 0), (1, 1), (1, 2), (1, 4), (2, 5), (3, 3), (4, 0), (4, 2), (4, 4)}
eps_ = 0
env = GridWorld(size=size_, battery_pos=battery_pos_, obstacles=obstacles_)
        
# initialize state
s0 = (0,2)
# set policy
pi = { s: [.25, .25, .25, .25] for s in env.states() }
gam_0_9 = 0.9
alpha = 0.1
eps = 0.1
n_samples = 200
T = 100

    
Q_SARSA, pi_SARSA = SARSA(env, gam_0_9, alpha, eps, s0, n_samples, T)
Q_q_learning, pi_q_learing = Q_learning(env, gam_0_9, alpha, eps, s0, n_samples, T)

V_SARSA = solve_Bellman_expectation(env, pi_SARSA, gam_0_9)
V_q_learing = solve_Bellman_expectation(env, pi_q_learing, gam_0_9)

plot_V(env, V_SARSA)
plot_V(env, V_q_learing)

print("Experiment finished!")