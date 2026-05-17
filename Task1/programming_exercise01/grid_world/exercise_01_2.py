# import grid world
from gridworld import GridWorld

# initialize grid world environment with given size, battery position and obstacle positions
N = 6
battery_pos_ = (4,3)
# obstacles_ = {(0, 0), (1, 1), (1, 2), (1, 4), (2, 5), (3, 3), (4, 0), (4, 2), (4, 4)}
obstacles_ = {(0, 0)}
env = GridWorld(size=N, battery_pos=battery_pos_, obstacles=obstacles_)
        
# initialize state
state = (4,2)

# test render function
env.render(state)

state = env.step(state, 'Right')
print(env.p(state, (4,2), 'Right'))
env.render(state)
state = env.step(state, 'Right')
print(env.p(state, (4,3), 'Right'))
env.render(state)
state = env.step(state, 'Left')
print(env.p(state, (4,3), 'Left'))
env.render(state)