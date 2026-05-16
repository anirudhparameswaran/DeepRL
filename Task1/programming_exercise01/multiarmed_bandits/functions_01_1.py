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
        #self.Mus = ?       # means
        self.sigma = 1.0     # uniform variance
        #self.K = ?         # number of arms
        #self.mu_star = ?   # optimal mean
        #self.a_star = ?    # optimal action

    def sample(self, a):
        return 0

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

    # ... 

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

    # ... 

    return avg_rewards, optimal_percentage, regret