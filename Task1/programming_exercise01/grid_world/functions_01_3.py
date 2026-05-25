import copy
import numpy as np
from collections import defaultdict
import random as random
import matplotlib.pyplot as plt
from pathlib import Path

def solve_Bellman_expectation(env, pi, gamma):
    """
    Solves the Bellman expectation equation directly via linear algebra:
    V^pi = (I - gamma P_pi)^(-1) R_pi
    """

    # Get state space and number of states
    states = sorted(list(env.states()))
    n_S = len(states)

    # store state values in an np.array
    V_pi = np.zeros(n_S)

    r_S = np.zeros(n_S)
    P = np.zeros((n_S, n_S))

    state_to_idx = {state: idx for idx, state in enumerate(states)}

    for s_i in states:
        i = state_to_idx[s_i]

        policy = pi[s_i]
        for a_idx, action in enumerate(env.actions):
            action_prob = policy[a_idx]

            if action_prob > 0.0:
                for s_j in states:
                    trans_prob = env.p(s_j, s_i, action)

                    if trans_prob > 0.0:
                        j = state_to_idx[s_j]
                        reward = env.reward(s_j, s_i, action)
                        r_S[i] += action_prob * trans_prob * reward
                        P[i, j] += action_prob * trans_prob
        
    # Solve for V^pi
    I = np.eye(n_S)
    A = I - gamma * P
    V_pi = np.linalg.solve(A, r_S)

    # Convert back to dictionary keyed by states
    return {states[i] : float(V_pi[i]) for i in range(n_S)}

def iterative_policy_evaluation(env, pi, gamma, max_iters=100, tol = 0, V_true=None):
    """
    Evaluate a fixed policy using iterative Bellman expectation updates.

    The value function is updated according to

        V_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)
                     [R(s,a,s') + γ V_k(s')]

    until convergence or until `max_iters` iterations are reached.

    Parameters
    ----------
    env : Environment
        Environment with transition probabilities, rewards, and successor states.

    pi : dict
        Policy where `pi[s][a]` is the probability of taking action `a`
        in state `s`.

    gamma : float
        Discount factor.

    max_iters : int, optional
        Maximum number of iterations.

    tol : float, optional
        Convergence threshold based on
        `max_s |V_new(s) - V_old(s)|`.

    V_true : dict, optional
        True value function used to track convergence error over iterations.

    Returns
    -------
    V_new : dict
        Estimated state-value function.

    dist : np.ndarray or None
        Per-iteration error relative to `V_true`, if provided.
    """
    
    # If true values are provided, track L1 error over iterations
    if V_true is None:
        dist = None
    else:
        dist = np.zeros(max_iters)

    # Initialize first iteration baseline
    V_last = {s : 0.0 for s in env.states()}

    for i in range(max_iters):

        # Initialize new value function
        V_new = {s : 0.0 for s in env.states()}


        # Bellman expectation update
        for s in env.states():
            # MAKE CHANGES HERE
            v_new = 0.0
            policy = pi[s]
            for a_idx, action in enumerate(env.actions):
                action_prob = policy[a_idx]

                if action_prob > 0.0:
                    for s_next in env.successors(s):
                        trans_prob = env.p(s_next, s, action)

                        if trans_prob > 0.0:
                            reward = env.reward(s_next, s, action)
                            v_new += action_prob * trans_prob * (reward + gamma * V_last[s_next])
                
            V_new[s] = v_new

        # stropping condition
        if max(np.abs(V_last[s] - V_new[s]) for s in env.states()) < tol:
            # Optional: compute distance to true value function
            if V_true is not None:
                dist[i] = max(np.abs(V_last[s] - V_true[s]) for s in env.states())

            dist = dist[0:i]
            return V_new, dist

        # Move updated values into "previous"
        V_last = V_new
        
        # Optional: compute distance to true value function
        if V_true is not None:
            dist[i] = max(np.abs(V_last[s] - V_true[s]) for s in env.states())

    return V_new, dist

def plot_V(env, V):
    '''
    Plots the given values for each state as a heatmap.
    If a state is not present in the values dict, it is assumed to have value 0.
    values: dict mapping (x,y) state tuples to scalar values
    '''
    
    img = np.zeros((env.size, env.size))
    for s in env.states():
        img[s[0], s[1]] = V.get(s, 0.0)
    
    plt.rcParams.update({
        "figure.figsize": (6.25, 5),
        "font.family": "serif",
        "mathtext.fontset": "cm",   # Computer Modern
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 3,
        "axes.grid": False,
        "grid.alpha": 0.3,
    })

    plt.imshow(img, cmap='hot', interpolation="nearest")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Value function $V(s)$')
    plt.xticks([0, 1, 2, 3])
    plt.yticks([0, 1, 2, 3])
    plt.colorbar()
    plt.show()
