from gridworld import GridWorld
from functions_01_3 import solve_Bellman_expectation, iterative_policy_evaluation, plot_V
import matplotlib.pyplot as plt

# initialize size, battery position, obstacle positions, epsilon
size_ = 4
battery_pos_ = (3,3)
obstacles_ = {(1, 1), (1, 2), (2, 2)}
env = GridWorld(size=size_, battery_pos=battery_pos_, obstacles=obstacles_)
# initialize state
init_state = (0, 0)

V = {(0, 0) : 0.4, (0, 1) : 0.5, (0, 2) : 0.6, (0, 3) : 0.7, (1, 0) : 0.5, (1, 3) : 0.8, (2, 0) : 0.6, (2, 1) : 0.7, (2, 3) : 0.9, (3, 0) : 0.7, (3, 1) : 0.8, (3, 2) : 0.9, (3, 3) : 1.0}

down = [1.0, 0.0, 0.0, 0.0]
down_right = [0.5, 0.0, 0.0, 0.5]
    
pi_dr = {s : down_right for s in env.states()}
pi_d = {s : down for s in env.states()}

gam_0_9 = 0.9
gam_0_1 = 0.1

#V_pi_dr_gam_0_9 = solve_Bellman_expectation()
#V_pi_dr_gam_0_1 = solve_Bellman_expectation()
#V_pi_d_gam_0_9 = solve_Bellman_expectation()

#plot_V()
#plot_V()
#plot_V()

#V_ipe_0_9, dist_ipe_0_9 = iterative_policy_evaluation()
#V_ipe_0_1, dist_ipe_0_1 = iterative_policy_evaluation()
        
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

#plt.plot(dist_ipe_0_9, label='Iterative policy evaluation, $\gamma = 0.9$', color='orange')
#plt.plot(dist_ipe_0_1, label='Iterative policy evaluation, $\gamma = 0.1$', color='blue')
    
plt.yscale('log')
plt.xlabel(r'Number of iterations')
plt.ylabel(r'$\| V_n - V^\pi \|_\infty$')
plt.title(r'Convergence of state value estimates')
plt.legend()
plt.grid(True)
plt.show()
