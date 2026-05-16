# import numpy and random
import numpy as np
import random

class GridWorld:

    def __init__(self,
                 size,
                 battery_pos,
                 obstacles):
        '''
        Initializes a GridWorld environment.

        size: int, the size of the grid (size x size)
        battery_pos: (x,y) tuple, the position of the battery
        obstacles: list of (x,y) tuples, the positions of the obstacles
        '''
        self.actions = ['Down','Up','Left','Right']
        self.num_actions = len(self.actions)
        self.battery_pos = battery_pos
        self.obstacles = set(obstacles)        
        self.size = size
        self.num_obstacles = len(obstacles)


    def states(self):
        '''
        Returns a set of all valid states in the gridworld (i.e., all positions that are not obstacles).
        '''
        states = []
        return states

    
    def successors(self, state):
        '''
        Returns a set of valid successor states that can be reached from the given state by taking any of the possible actions.
        I.e., all states s' with sum_a p(s'|s,a) > 0.
        '''

        return {}

    def p(self, successor, state, action):
        '''
        Returns the probability of transitioning from state to successor
        using the given action.
        '''

        return 0 

    def step(self, state, action):
        '''
        Given a state and an action, returns the successor for the transition under action. 
        '''        
        succs = state

        return succs
    
    def reward(self, successor, state, action):
        '''
        Returns the reward for transitioning from state to successor
        using the given action.
        '''

        r = 0.0

        return r
    
    def render(self, state):
        '''
        Prints the gridworld to the console, with the agent's current position marked as 'X', 
        obstacles marked as '#', and the battery marked as 'B'.
        '''
        grid = np.full((self.size,self.size), '_')
        print("\n")
        for obs in self.obstacles: 
            grid[*obs] = '#'
        grid[*self.battery_pos] = 'B'
        grid[*state] = 'X'
        print("\n".join("".join(row) for row in grid))
        print("\n")