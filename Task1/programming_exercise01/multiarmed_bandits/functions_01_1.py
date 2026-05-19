# import numpy
import numpy as np

class MultiArmedBandit:
    """
    Represents a multi-armed bandit environment.

    Each arm has its own reward distribution with:
    - a mean value (stored in mus)
    - a fixed variance (sigma)

    The agent can choose an arm/action and receive
    a randomly sampled reward.
    """

    def __init__(self, mus):
        self.rng = np.random.default_rng()  # random number generator
        self.Mus = mus      # means
        self.sigma = 1.0     # uniform variance
        self.K = 10         # number of arms
        self.mu_star = np.max(self.Mus)   # optimal mean
        self.a_star = np.argmax(self.Mus)    # optimal action

    def sample(self, a):
        return self.rng.normal(self.Mus[a], self.sigma)

def epsilon_greedy(multiArmedBandit, n=10000, epsilon=0.1, time_varying=False, delta=1):
    """
    Epsilon-greedy algorithm for the multi-armed bandit problem.

    Balances exploration (random actions) and exploitation
    (best estimated action), while tracking rewards,
    optimal-action rate, and cumulative regret.

    Parameters:
    -----------
    multiArmedBandit : MultiArmedBandit
        Bandit environment.

    n : int
        Number of iterations.

    epsilon : float
        Exploration probability.

    time_varying : bool
        If True, uses epsilon = delta / t.

    delta : float
        Decay constant for time-varying epsilon.

    Returns:
    --------
    rewards : list
    optimal_percentage : list
    regret : list
    """

    avg_rewards = []
    optimal_percentage = []
    regret = []

    Q = np.zeros(multiArmedBandit.K)
    N = np.zeros(multiArmedBandit.K)

    total_reward = 0.0
    optimal_action_count = 0
    cumulative_regret = 0.0

    for i in range(n):
        if time_varying:
            epsilon = delta / (i+1)

        if np.random.rand() < epsilon:
            action = multiArmedBandit.rng.integers(multiArmedBandit.K)  # Explore: random action
        else:
            action = np.argmax(Q)  # Exploit: best estimated action
        
        reward = multiArmedBandit.sample(action)  # Sample reward
        total_reward += reward
        
        if action == multiArmedBandit.a_star:
            optimal_action_count += 1
        cumulative_regret += (multiArmedBandit.mu_star - multiArmedBandit.Mus[action])

        avg_rewards.append(total_reward / (i + 1))
        optimal_percentage.append(optimal_action_count / (i + 1))
        regret.append(cumulative_regret)

        N[action] += 1  # Update count for the chosen action
        Q[action] += (reward - Q[action]) / N[action] 


    return avg_rewards, optimal_percentage, regret

def ucb(multiArmedBandit, optimal_arm, n=10000, c=2):
    """
    Upper Confidence Bound (UCB) algorithm for the
    multi-armed bandit problem.

    Selects actions using estimated rewards and an
    exploration bonus to balance exploration and exploitation,
    while tracking rewards, optimal-action rate,
    and cumulative regret.

    Parameters:
    -----------
    multiArmedBandit : MultiArmedBandit
        Bandit environment.

    n : int
        Number of iterations.

    c : float
        Exploration coefficient controlling confidence bounds.

    Returns:
    --------
    rewards : list
    optimal_percentage : list
    regret : list
    """

   
    avg_rewards = []
    optimal_percentage = []
    regret = []

    Q = np.zeros(multiArmedBandit.K)
    N = np.zeros(multiArmedBandit.K)

    total_reward = 0.0
    optimal_action_count = 0
    cumulative_regret = 0.0

    for t in range(1, n+1):
        ucb_values = np.zeros(multiArmedBandit.K)
        for a in range(multiArmedBandit.K):
            if N[a] == 0:
                ucb_values[a] = np.inf  # If N_t-1(a) == 0, value is +infinity
            else:
                ucb_values[a] = Q[a] + c * np.sqrt(np.log(t) / N[a])
        
        action = np.argmax(ucb_values)

        reward = multiArmedBandit.sample(action)
        total_reward += reward

        if action == multiArmedBandit.a_star:
            optimal_action_count += 1
        cumulative_regret += (multiArmedBandit.mu_star - multiArmedBandit.Mus[action])

        avg_rewards.append(total_reward / (t))
        optimal_percentage.append(optimal_action_count / (t))
        regret.append(cumulative_regret)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

    return avg_rewards, optimal_percentage, regret