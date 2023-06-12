from math import tau
from operator import index
import constants as c
import numpy as np
from tiles3 import IHT, tileswrap


class Environ(object):
    def __init__(self, agent) -> None:
        self.__q = None
        self.__dq = None
        self.__reward = None
        self.__num_features = c.NUM_FEATURES
        self.__num_tilings = c.NUM_TILINGS
        self.__num_actions = c.NUM_ACTIONS
        self.agent = agent
        self.__alpha: float = c.ALPHA
        self.__gamma: float = c.GAMMA
        self.__sigma: float = c.SIGMA
        self.__epsilon: float = c.EPSILON
        self.__num_episodes = c.NUM_EPISODES
        self.__features = np.zeros((c.NUM_FEATURES, 1))  # features vector
        self.__w = np.zeros((c.NUM_ACTIONS, c.NUM_FEATURES))  # weights matrix
        self.__q1b, self.__q2b = c.WORKING_RANGE  # lower and upper bounds for q
        self.__dq1b, self.__dq2b = c.WORKING_VELOCITIES  # lower and upper bounds for dq
        self.__widths = np.array(
            [
                self.__q1b[1] - self.__q1b[0],
                self.__dq1b[1] - self.__dq1b[0],
                self.__q2b[1] - self.__q2b[0],
                self.__dq2b[1] - self.__dq2b[0],
            ]
        )  # widths of the working range
        self.iht = IHT(c.NUM_FEATURES)  # index hash table
        self.__reward_history = np.zeros((self.__num_episodes, 1))
        self.__action_history = [[]] * self.__num_episodes

    def get_info(self):
        print(
            f"Number of features: {self.__num_features}\n"
            + f"Number of actions: {self.__num_actions}\n"
            + f"Number of weights (totals): {np.cumprod(self.__w.shape)[-1]}\n"
            f"Number of episodes: {self.__num_episodes}\n"
            + f"Number of tiles per dimension: {self.__num_tilings}\n"
            + f"Bounds for q: {self.__q1b, self.__q2b}\n"
            + f"Bounds for dq: {self.__dq1b, self.__dq2b}"
        )

    def get_features(self, state):
        """
        Returns the features vector for the current state.
        """
        # get the indices of the active tiles
        active_tiles = self.get_indexes(state)
        # reset the features vector
        self.__features[:] = 0
        # set the active features to 1
        self.__features[active_tiles] = 1
        return self.__features

    def epsGreedy(self, state):
        """
        Returns the action according to the epsilon greedy policy.
        The actions are chosen according to the weights of the RBF and represents
        the torque applied to the system, tau1 and tau2.
        """
        if np.random.rand() < self.__epsilon:
            # return a random action with probability epsilon
            a_i = np.random.randint(0, self.__num_actions)
        else:
            # approximate the Q function with the weights
            q_a = self.__w @ self.get_features(state)
            # return the action with the highest Q value
            a_i = np.argmax(q_a)

        return self.agent.get_action(a_i), a_i

    def get_indexes(self, state):
        """
        Returns the tiles of the current state.
        """

        q1, dq1, q2, dq2 = state

        scaleFactor_q1 = self.__num_tilings / (self.__widths[0])
        scaleFactor_dq1 = self.__num_tilings / (self.__widths[1])
        scaleFactor_q2 = self.__num_tilings / (self.__widths[2])
        scaleFactor_dq2 = self.__num_tilings / (self.__widths[3])

        return tileswrap(
            self.iht,  # index hash table
            self.__num_tilings,  # num tilings
            [  # coordinates of the state
                scaleFactor_q1 * q1,
                scaleFactor_dq1 * dq1,
                scaleFactor_q2 * q2,
                scaleFactor_dq2 * dq2,
            ],
            # width of each tiling
            [self.__widths[0], self.__widths[1], self.__widths[2], self.__widths[3]],
        )

    def is_done(self):
        """
        Checks if the current state is terminal.
        """
        # compute the forward kinematics to get the position
        # of the end effector at the current state
        p = self.agent.get_p()
        if self.agent._is_inside_ws(p):
            if np.linalg.norm(p - self.agent.target) ** 2 <= self.__epsilon:
                return True
        return False

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
            self.__action_history[episode] = [index]
            # Run the episode until the agent reaches the goal
            step = 0
            ti = 0
            while not self.is_done():
                # Update the step and time variables
                step += 1
                tf = ti + c.TIME_STEP
                #print(
                #    f"step: {step}\t reward: {self.__reward_history[episode]}\t"
                #    + f"pos: {self.agent.get_p()}\t ws: {self.agent._is_inside_ws(self.agent.get_p())}\t"
                #    + f"q: {self.agent.get_q()}\t dq: {self.agent.get_dq()}"
                #)
                # Get esteem for the current state
                x = self.get_features(state)
                # Get the next state
                next_state = self.agent.get_next_state(state, action, [ti, tf])
                # Get the next action
                next_action, next_index = self.epsGreedy(next_state)
                self.__action_history[episode].append(next_index)
                # Get the reward for the current state
                reward = self.agent.get_reward(episode)
                # Update the reward history for the current episode
                self.__reward_history[episode] += reward
                # Verify if the next state is terminal to skip useless computations
                if self.is_done():
                    # Compute the TD error
                    delta = reward - self.__w @ x
                    # Update the weights
                    self.__w[index] += self.__alpha * delta * x
                    print("Episode {episode} finished after {} timesteps".format(step))
                    break
                # Get esteem for the next state
                xp = self.get_features(next_state)
                # Compute the TD error for the next state
                delta = reward + self.__gamma * self.__w @ xp - self.__w @ x
                # Update the weights
                self.__w[next_index] += self.__alpha * delta[next_index] * x.T[0]
                # Set new values for the state and action
                state = next_state
                action = next_action
                ti = tf
