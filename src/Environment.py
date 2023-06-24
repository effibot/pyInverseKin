from math import tau
from operator import index
import constants as c
import numpy as np
from tiles3 import IHT, tiles
from agent import agent
from numpy.linalg import norm
import pickle


class Environ(object):
    def __init__(self, target) -> None:
        self.__num_features = c.NUM_FEATURES
        self.__num_tilings = c.NUM_TILINGS
        self.__num_actions = c.NUM_ACTIONS
        self.agent = agent(target)
        self.__alpha: float = c.ALPHA
        self.__gamma: float = c.GAMMA
        self.__epsilon: float = c.EPSILON
        self.__num_episodes = c.NUM_EPISODES
        self.__features = np.zeros((c.NUM_FEATURES, 1))  # features vector
        self.__w = np.zeros((c.NUM_ACTIONS, c.NUM_FEATURES))  # weights matrix
        self.__widths = np.array(
            [
                c.WORKING_RANGE[0, 1] - c.WORKING_RANGE[0, 0],
                c.WORKING_RANGE[1, 1] - c.WORKING_RANGE[1, 0],
            ]
        )  # widths of the working range
        self.iht = IHT(c.NUM_FEATURES)  # index hash table

    def get_info(self):
        print(
            f"Number of features: {self.__num_features}\n"
            + f"Number of actions: {self.__num_actions}\n"
            + f"Number of weights (totals): {np.cumprod(self.__w.shape)[-1]}\n"
            f"Number of episodes: {self.__num_episodes}\n"
            + f"Number of tiles per dimension: {self.__num_tilings}\n"
            + f"Bounds for q: {c.WORKING_RANGE[0,:], c.WORKING_RANGE[1,:]}\n"
            + f"Bounds for dq: {c.WORKING_VELOCITIES[0], c.WORKING_VELOCITIES[1]}\n"
        )

    def get_features(self, state, action):
        """
        Returns the features vector for the current state.
        """
        # get the indices of the active tiles
        active_tiles = self.get_indexes(state, action)
        # reset the features vector
        self.__features[:] = 0
        # set the active features to 1
        self.__features[active_tiles] = 1
        return self.__features

    def epsGreedy(self, state, action=None):
        """
        Returns the action according to the epsilon greedy policy.
        The actions are chosen according to the weights of the RBF and represents
        the torque applied to the system, tau1 and tau2.
        """
        if np.random.rand() < self.__epsilon or action is None:
            # return a random action with probability epsilon
            a_i = np.random.randint(0, self.__num_actions)
        else:
            # approximate the Q function with the weights
            q_a = self.__w @ self.get_features(state, action)
            # return the action with the highest Q value
            a_i = np.argmax(q_a)

        return self.agent.get_action(a_i), a_i

    def get_indexes(self, state, action):
        """
        Returns the tiles of the current state.
        """

        q1, q2, dq1, dq2 = state

        scaleFactor_q1 = self.__num_tilings / np.abs(self.__widths[0])
        scaleFactor_q2 = self.__num_tilings / np.abs(self.__widths[1])
        scaleFactor_dq1 = self.__num_tilings
        scaleFactor_dq2 = self.__num_tilings

        return tiles(
            ihtORsize=self.iht,  # index hash table
            numtilings=self.__num_tilings,  # num tilings
            floats=[  # coordinates of the state
                scaleFactor_q1 * q1,
                scaleFactor_q2 * q2,
                scaleFactor_dq1 * dq1,
                scaleFactor_dq2 * dq2,
            ],
            ints=action,
        )

    def SARSA(self):
        """
        This function implements the SARSA algorithm for the double pendulum system.
        """
        for episode in range(0, self.__num_episodes):
            print("Episode: ", episode)
            # Reset the environment for new episode
            self.agent.reset()
            # Get the initial state
            state = self.agent.get_state()
            # Get the initial action
            action, index = self.epsGreedy(state)
            # Run the episode until the agent reaches the goal
            step = 0
            # while step < 20000:
            while not self.agent.is_done() and step < c.MAX_STEPS:
                self.agent.set_pos_history(step)
                # Update the step and time variables
                step += 1
                # print(
                #     f"step: {step}\t reward: {self.agent.reward_history[episode]}\t"
                #     + f"pos: {self.agent.get_p()}\t ws: {self.agent.is_inside_ws()}\t"
                #     + f"q: {self.agent.get_state()[0:2]}\t dq: {self.agent.get_state()[2:]}"
                # )
                # Get esteem for the current state
                x = self.get_features(state, action)
                # Get the next state and the reward
                next_state, reward = self.agent.get_state_reward_transition(
                    state, action, episode
                )
                # Get the next action
                next_action, next_index = self.epsGreedy(next_state, action)
                # Verify if the next state is terminal to skip useless computations
                if self.agent.is_done():
                    self.agent.set_pos_history(step)
                    # Compute the TD error
                    delta = reward - self.__w @ x
                    # Update the weights
                    self.__w[index] += self.__alpha * delta * x
                    pickle.dump(self.__w, open("weights.pkl", "wb"))
                    pickle.dump(x, open("features.pkl", "wb"))
                    print("Episode {episode} finished after {} timesteps".format(step))
                    break
                # Get esteem for the next state
                xp = self.get_features(next_state, next_action)
                # Compute the TD error for the next state
                delta = reward + self.__gamma * self.__w @ xp - self.__w @ x
                # Update the weights
                self.__w[next_index] += self.__alpha * delta[next_index] * x.T[0]
                # Set new values for the state and action
                state = next_state
                action = next_action
            if step >= 3000:
                break
