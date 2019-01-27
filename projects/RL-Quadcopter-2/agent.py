import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def epsilon_greedy_probs(self,Q_s, i_episode, eps=None):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = 0.5 / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def select_action(self,state, ep):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        import pdb;pdb.set_trace()
        policy_s = self.epsilon_greedy_probs(self.Q[state], ep)
        return np.random.choice(self.nA,p=policy_s)

    def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        alpha=1
        gamma=0.1
        Qsa = self.Q[state][action]
        Qsa_next = np.max(self.Q[next_state])
        self.Q[state][action] = Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))
        return self.Q[state][action]
