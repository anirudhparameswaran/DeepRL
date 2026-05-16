import numpy as np
from collections import defaultdict
import random as random
import random


def td_prediction(env, pi, gamma, alpha, s0, n_samples=200, T=100, V_true = None):
    """
    Estimate the state-value function V^π using the TD(0) algorithm.

    This function performs Monte Carlo simulation of episodes under policy π,
    and applies temporal-difference learning to incrementally update value
    estimates for each state.

    Parameters
    ----------
    env : object
        Environment object that defines:
        - env.states(): iterable of all states
        - env.num_actions: number of available actions
        - env.step(s, a): transition function returning next state
        - env.reward(s', s, a): reward function
        - env.battery_pos: terminal or absorbing state indicator

    pi : dict or array-like
        Policy π(a|s), where pi[s] provides a probability distribution over actions.

    gamma : float
        Discount factor in [0, 1], determining importance of future rewards.

    alpha : float
        Base learning rate for TD updates (decayed as α / sqrt(i + 1)).

    s0 : state or None
        Initial state for each episode. If None, a state is sampled uniformly
        from env.states().

    n_samples : int, default=200
        Number of episodes used for learning.

    T : int, default=100
        Maximum number of time steps per episode.

    V_true : dict or None, default=None
        Optional ground-truth value function used for evaluation of learning
        performance. If provided, L∞ error over states is tracked.

    Returns
    -------
    V : defaultdict(float)
        Learned approximation of the state-value function V^π.

    dist : np.ndarray or None
        Array of shape (n_samples,) containing the max absolute error
        ||V - V_true||_∞ after each episode, if V_true is provided;
        otherwise None.
    """
    # Value estimate for each state
    V = defaultdict(float)

    # Optional tracking of L1 error vs true value function
    if V_true is None:
        dist = None
    else:
        dist = np.zeros(n_samples)

    # Loop over episodes
    for i in range(n_samples):
        
        # MAKE CHANGES HERE
        
        # Optional diagnostics
        if V_true is not None:
            dist[i] = max(np.abs(V[s] - V_true[s]) for s in env.states())

    return V, dist

def SARSA(env, gamma, alpha, eps, s0, n_samples=200, T=100):
    """
    Estimate the action-value function Q^π using the on-policy SARSA algorithm
    with ε-greedy exploration.

    The agent interacts with the environment by following an ε-greedy policy
    derived from the current action-value estimates. After each transition
    (s, a, r, s', a'), the Q-values are updated using the SARSA update rule:

        Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') − Q(s, a)]

    The policy π is updated online to remain ε-greedy with respect to Q.

    Parameters
    ----------
    env : object
        Environment providing:
        - env.states(): iterable of all states
        - env.actions: list of available actions
        - env.num_actions: number of actions
        - env.step(s, a): transition function returning next state
        - env.reward(s', s, a): reward function
        - env.battery_pos: terminal state indicator

    gamma : float
        Discount factor in [0, 1], controlling importance of future rewards.

    alpha : float
        Learning rate used in TD updates.

    eps : float
        Exploration rate for ε-greedy policy.

    s0 : state or None
        Initial state for each episode. If None, states are sampled uniformly.

    n_samples : int, default=200
        Number of episodes used for learning.

    T : int, default=100
        Maximum number of time steps per episode.

    Q_true : dict or None, default=None
        Optional ground-truth action-value function used for evaluation.
        If provided, performance is tracked using the maximum absolute error.

    Returns
    -------
    Q : dict
        Learned action-value function mapping (state, action) pairs to values.

    pi : dict
        Final ε-greedy policy derived from Q, where pi[s] is a probability
        distribution over actions.

    dist : np.ndarray or None
        Array of shape (n_samples,) containing the max-norm error
        ||Q - Q_true||_∞ after each episode, if Q_true is provided;
        otherwise None.
    """
    # Value estimate for each state
    Q = {(s, a): 0.0 for s in env.states() for a in range(len(env.actions))}

    pi = compute_eps_greedy_policy(env, Q, eps)

    # Loop over Monte Carlo episodes
    for i in range(n_samples):
        # MAKE CHANGES HERE
        Q = Q

    return Q, pi

def Q_learning(env, gamma, alpha, eps, s0, n_samples=200, T=100):
    """
    Estimate the optimal action-value function Q* using the Q-learning algorithm
    with ε-greedy exploration.

    This implementation performs off-policy temporal-difference learning.
    After each transition (s, a, r, s'), the action-value function is updated
    using the Bellman optimality target:

        Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') − Q(s, a)]

    The behavior policy is ε-greedy with respect to the current Q-values and
    is updated online throughout training.

    Parameters
    ----------
    env : object
        Environment providing:
        - env.states(): iterable of all states
        - env.actions: list of available actions
        - env.num_actions: number of actions
        - env.step(s, a): transition function returning next state
        - env.reward(s', s, a): reward function
        - env.battery_pos: terminal state indicator

    gamma : float
        Discount factor in [0, 1], controlling importance of future rewards.

    alpha : float
        Learning rate used in TD updates.

    eps : float
        Exploration rate for ε-greedy behavior policy.

    s0 : state or None
        Initial state for each episode. If None, states are sampled uniformly.

    n_samples : int, default=200
        Number of episodes used for learning.

    T : int, default=100
        Maximum number of time steps per episode.

    Q_true : dict or None, default=None
        Optional ground-truth optimal action-value function used for evaluation.
        If provided, performance is tracked using the maximum absolute error.

    Returns
    -------
    Q : dict
        Learned action-value function approximating Q*.

    pi : dict
        Final ε-greedy policy derived from Q, mapping each state to a
        probability distribution over actions.

    dist : np.ndarray or None
        Array of shape (n_samples,) containing the max-norm error
        ||Q - Q_true||_∞ after each episode, if Q_true is provided;
        otherwise None.
    """
    # Value estimate for each state
    Q = {(s, a): 0.0 for s in env.states() for a in range(len(env.actions))}

    pi = compute_eps_greedy_policy(env, Q, eps)

    # Loop over Monte Carlo episodes
    for i in range(n_samples):
        # Initialize episode starting state
        state = s0 if s0 is not None else random.choice(list(env.states()))
        
        # Generate an episode of fixed horizon T
        for t in range(T+1):
            # Sample action from policy π(a|s)
            action = random.choices(range(env.num_actions), weights=pi[state])[0]
            # Environment transition
            next_state = env.step(state, action)
            # Reward received from transition
            reward = env.reward(next_state, state, action)
            # Check if next state is terminal
            if next_state == env.battery_pos:
                Q[state, action] += alpha*(reward - Q[state, action])
                break
            # Update value estimate using temporal differences
            Q[state, action] += alpha*(reward + gamma*max(Q[next_state, a] for a in range(len(env.actions))) - Q[state, action])
            
            # Update policy
            Q_s = np.array([Q[(state, a)] for a in range(len(env.actions))])
            max_Q_s = np.max(Q_s)
            greedy_indices = np.where(Q_s >= max_Q_s - 1e-12)[0]
            n_greedy = len(greedy_indices)
            # base exploration probability
            pi[state][:] = eps / len(env.actions)
            # add exploitation mass to greedy actions
            for a in greedy_indices:
                pi[state][a] += (1.0 - eps) / n_greedy
            
            state = next_state

    return Q, pi

def compute_eps_greedy_policy(env, Q, eps):

    pi = {s: np.zeros(len(env.actions)) for s in env.states()}
    tol = 1e-6

    nA = len(env.actions)

    for s in env.states():
        Q_s = np.array([Q[(s, a)] for a in range(nA)])
        max_Q_s = np.max(Q_s)

        greedy_indices = np.where(Q_s >= max_Q_s - tol)[0]
        n_greedy = len(greedy_indices)

        # base exploration probability
        pi[s][:] = eps / nA

        # add exploitation mass to greedy actions
        for a in greedy_indices:
            pi[s][a] += (1.0 - eps) / n_greedy

    return pi