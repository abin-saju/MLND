from agent import Agent
from monitor import interact
import gym
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from task import Task

env = Task(target_pos = [0.,0.,10.])

#env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
plt.plot(np.exp(avg_rewards))
plt.show()
