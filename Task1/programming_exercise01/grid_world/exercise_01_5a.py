from gridworld import GridWorld
from functions_01_3 import solve_Bellman_expectation
from functions_01_4 import ev_monte_carlo, fv_monte_carlo
from functions_01_5 import td_prediction
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
np.random.seed(42)

    
# initialize size, battery position, obstacle positions, epsilon
size_ = 4
battery_pos_ = (3,3)
obstacles_ = {(1, 1), (1, 2), (2, 2)}
eps_ = 0
env = GridWorld(size=size_, battery_pos=battery_pos_, obstacles=obstacles_)
# initialize state
init_state = (0, 0)

down_right = [0.5, 0.0, 0.0, 0.5]
    
pi = {s : down_right for s in env.states()}
gam_0_9 = 0.9
V_pi = solve_Bellman_expectation(env, pi, gam_0_9)
    
n_runs = 20
all_dist_ev_mc = []
all_dist_fv_mc = []
all_dist_td = []

s0 = init_state

for run in tqdm(range(n_runs), desc="Running MC experiments"):
          
    _, dist_ev_mc = ev_monte_carlo(env, pi, gam_0_9, s0=s0, n_samples=2000, T=128, V_true = V_pi)
    all_dist_ev_mc.append(dist_ev_mc)

    _, dist_fv_mc = fv_monte_carlo(env, pi, gam_0_9, s0=s0, n_samples=2000, T=128, V_true = V_pi)
    all_dist_fv_mc.append(dist_fv_mc)

    _, dist_td = td_prediction(env, pi, gam_0_9, 1, s0 = s0, n_samples=2000, T=128, V_true = V_pi)
    all_dist_td.append(dist_td)

# Compute mean and standard deviation
mean_ev_mc = np.mean(all_dist_ev_mc, axis=0)
std_ev_mc = np.std(all_dist_ev_mc, axis=0)
mean_fv_mc = np.mean(all_dist_fv_mc, axis=0)
std_fv_mc = np.std(all_dist_fv_mc, axis=0)
mean_td = np.mean(all_dist_td, axis=0)
std_td = np.std(all_dist_td, axis=0)

# Plot with shaded variance
plt.figure(figsize=(8,5))

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Computer Modern
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 3,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

plt.plot(mean_ev_mc, label='Every visit Monte-Carlo method', color='blue')
plt.fill_between(np.arange(len(mean_ev_mc)), mean_ev_mc - std_ev_mc, mean_ev_mc + std_ev_mc, color='blue', alpha=0.3)

plt.plot(mean_fv_mc, label='First visit Monte-Carlo method', color='orange')
plt.fill_between(np.arange(len(mean_fv_mc)), mean_fv_mc - std_fv_mc, mean_fv_mc + std_fv_mc, color='orange', alpha=0.3)

plt.plot(mean_td, label='Temporal difference learning', color='green')
plt.fill_between(np.arange(len(mean_td)), mean_td - std_td, mean_td + std_td, color='green', alpha=0.3)

plt.yscale('log')
plt.xlabel(r'Number of iterations')
plt.ylabel(r'$\| V_n - V^\pi \|_\infty$')
plt.title(r'Convergence of state value estimates')
plt.legend()
plt.grid(True)
plt.show()

print("Experiment finished!")