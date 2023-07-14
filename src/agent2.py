import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, ndarray, sin
from numpy.linalg import norm

import constants as c
from tiles3 import IHT, tiles


class agent:
    def __init__(self, target, env=None, exploring_start=False):
        self.constants = {"l1": c.L1, "l2": c.L2, "m1": c.M1, "m2": c.M2, "g": c.g}
        self.is_sat = True
        self.target: ndarray = target
        self.achieved_target = False
        self.exploring_start = exploring_start
        self.state = c.init_cond
        self.time_step = c.TIME_STEP
        self.env = env
        self.iht = IHT(c.NUM_FEATURES)
        self.actions: np.ndarray = [1e5, 1e4] * np.asarray(
            (
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            )
        )
        self.actions_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS + 1, 2))
        self.reward_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS + 1))
        self.position_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS + 1, 2))
        self.ws = self.generate_ws_curves()

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def set_env(self, env):
        self.env = env

    def get_env(self):
        return self.env

    def get_action(self, index):
        return self.actions[index]

    def get_achieved_target(self):
        return self.achieved_target

    def set_achieved_target(self, done):
        self.achieved_target = done

    def set_pos_history(self, episode, step, pos):
        self.position_history[episode, step] = pos

    def set_action_history(self, episode, step, action):
        self.actions_history[episode, step] = action

    def set_reward_history(self, episode, step, reward):
        self.reward_history[episode, step] = reward

    def update_histories(self, episode, step, action, reward, pos):
        self.set_action_history(episode, step, action)
        self.set_reward_history(episode, step, reward)
        self.set_pos_history(episode, step, pos)

    def get_histories(self):
        return {
            "position": self.position_history,
            "actions": self.actions_history,
            "reward": self.reward_history,
        }

    def reset(self):
        self.set_state(c.init_cond)
        return self.get_state()

    def h_q(self, q1, q2):
        """Compute the forward kinematics of the robot.

        Args:
            q1: The angle of the first joint. Rad.
            q2: The angle of the second joint. Rad.

        Returns:
            A tuple containing the x and y coordinates of the end effector.
        """
        x1 = c.L1 * cos(q1)
        y1 = c.L1 * sin(q1)
        x2 = c.L1 * cos(q1) + c.L2 * cos(q1 + q2)
        y2 = c.L1 * sin(q1) + c.L2 * sin(q1 + q2)
        return np.asarray([x1, y1, x2, y2])

    def is_inside_ws(self, x, y):
        """Check if the current state is inside the workspace of the robot."""
        cond = ((x**2 + y**2) < (c.L1 + c.L2) ** 2) & ((x**2 + y**2) > (c.L1 - c.L2) ** 2)
        return cond

    def generate_ws_curves(self, n=1000):
        """
        This function generates workspace curves by computing the forward kinematics for a grid of
        points within a defined range and filtering out points outside the workspace.

        :param n: The number of points to use to draw the workspace curves. The higher the number, the
        smoother the curve
        :return: two arrays, X and Y, which contain the coordinates of the points that define the
        workspace curves.
        """
        # define how many points to use to draw the curves. The higher the number, the smoother the curve
        n_points = n
        # define the range of angles to use
        t1 = np.linspace(c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1], n_points)[:-1]
        t2 = np.linspace(c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1], n_points)[:-1]
        # create the grid of points
        grid = np.meshgrid(t1, t2)
        # compute the forward kinematics for each point in the grid
        X, Y = self.h_q(grid[0], grid[1])
        # filter out the points that are outside the workspace
        X, Y = X[self.is_inside_ws(X, Y)], Y[self.is_inside_ws(X, Y)]
        # filter out duplicate of pairs of points
        X, Y = np.unique([X, Y], axis=1)
        # return the points
        return X, Y

    def compute_distance(self, state):
        """Compute the distance between the end effector and the target."""
        p = self.h_q(state[0], state[1])[2:]
        return norm(p - self.target)

    # define the vector field function
    def dxdt(self, x, tau):
        """This function computes the time derivative of the state vector x.

        Args:
            x (np.array): state vector
            constant (dictionary): constants dictionary for the system
            tau (list): torques

        Returns:
            np.array: time derivative of the state vector
        """
        # unpack the state vector
        x1, x2, x3, x4 = x
        # unpack the constants
        l1 = self.constants["l1"]
        l2 = self.constants["l2"]
        m1 = self.constants["m1"]
        m2 = self.constants["m2"]
        g = self.constants["g"]
        # unpack the torques
        t1, t2 = tau
        # compute the time derivative of the state vector
        dx1dt = x3
        dx2dt = x4
        dx3dt = (
            l2 * t1
            - l2 * t2
            - l2 * x3
            + l2 * x4
            - l1 * t2 * cos(x2)
            + l1 * x4 * cos(x2)
            + l1 * l2**2 * m2 * x3**2 * sin(x2)
            + l1 * l2**2 * m2 * x4**2 * sin(x2)
            + g * l1 * l2 * m1 * cos(x1)
            + g * l1 * l2 * m2 * cos(x1)
            - g * l1 * l2 * m2 * cos(x1 + x2) * cos(x2)
            + l1**2 * l2 * m2 * x3**2 * cos(x2) * sin(x2)
            + 2 * l1 * l2**2 * m2 * x3 * x4 * sin(x2)
        ) / (l1**2 * l2 * m1 + l1**2 * l2 * m2 - l1**2 * l2 * m2 * cos(x2) ** 2)
        dx4dt = -(
            l2**2 * m2 * t1
            - l1**2 * m2 * t2
            - l1**2 * m1 * t2
            - l2**2 * m2 * t2
            + l1**2 * m1 * x4
            + l1**2 * m2 * x4
            - l2**2 * m2 * x3
            + l2**2 * m2 * x4
            + l1 * l2**3 * m2**2 * x3**2 * sin(x2)
            + l1**3 * l2 * m2**2 * x3**2 * sin(x2)
            + l1 * l2**3 * m2**2 * x4**2 * sin(x2)
            - g * l1**2 * l2 * m2**2 * cos(x1 + x2)
            + g * l1 * l2**2 * m2**2 * cos(x1)
            + l1 * l2 * m2 * t1 * cos(x2)
            - 2 * l1 * l2 * m2 * t2 * cos(x2)
            - l1 * l2 * m2 * x3 * cos(x2)
            + 2 * l1 * l2 * m2 * x4 * cos(x2)
            + l1**3 * l2 * m1 * m2 * x3**2 * sin(x2)
            + 2 * l1 * l2**3 * m2**2 * x3 * x4 * sin(x2)
            - g * l1 * l2**2 * m2**2 * cos(x1 + x2) * cos(x2)
            + 2 * l1**2 * l2**2 * m2**2 * x3**2 * cos(x2) * sin(x2)
            + l1**2 * l2**2 * m2**2 * x4**2 * cos(x2) * sin(x2)
            + g * l1**2 * l2 * m2**2 * cos(x1) * cos(x2)
            - g * l1**2 * l2 * m1 * m2 * cos(x1 + x2)
            + g * l1 * l2**2 * m1 * m2 * cos(x1)
            + 2 * l1**2 * l2**2 * m2**2 * x3 * x4 * cos(x2) * sin(x2)
            + g * l1**2 * l2 * m1 * m2 * cos(x1) * cos(x2)
        ) / (-(l1**2) * l2**2 * m2**2 * cos(x2) ** 2 + l1**2 * l2**2 * m2**2 + m1 * l1**2 * l2**2 * m2)

        return np.asarray([dx1dt, dx2dt, dx3dt, dx4dt])

    # define the stap function to compute the next state using RK4 integration
    def step(self, x, tau):
        """This function computes the next state of the system with RK4 method.

        Args:
            x (np.array): state vector
            tau (list): torques

        Returns:
            np.array: next state of the system
        """
        # RK4 integration
        k1 = self.dxdt(x, tau)
        k2 = self.dxdt(x + self.time_step * k1 / 2, tau)
        k3 = self.dxdt(x + self.time_step * k2 / 2, tau)
        k4 = self.dxdt(x + self.time_step * k3, tau)
        # update the state
        next_state += self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # saturate
        if self.is_sat:
            # the first joint should be in the range [0, 2pi] but with saturation
            # there will be jumps in the in the dynamics. Leaving it free to move
            # in the range [-inf, inf] is better
            # self.state[0] = np.clip(self.state[0], c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1])
            next_state[1] = np.clip(next_state[1], c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1])
            next_state[2] = np.clip(next_state[2], c.WORKING_VELOCITIES[0, 0], c.WORKING_VELOCITIES[0, 1])
            next_state[3] = np.clip(next_state[3], c.WORKING_VELOCITIES[1, 0], c.WORKING_VELOCITIES[1, 1])
        return next_state

    # RL functions

    def get_reward(self, state, episode):
        """
        Returns the reward for the current state.
        """
        # compute the distance between the end effector and the target
        covered_distance = self.compute_distance(state)[2:]
        # compute the reward
        reward = -(covered_distance**2)
        # get last reward
        nonzero_index = np.nonzero(self.reward_history[episode])[0]
        if len(nonzero_index) > 0:
            if reward == self.reward_history[episode, nonzero_index[-1]]:
                reward *= 1.5
        # return the reward
        return reward

    def is_done(self):
        """
        Checks if the current state is terminal.
        """
        cond = self.compute_distance(self.state) <= c.GAMMA
        if cond:
            self.set_achieved_target(cond)
            return True
        self.set_achieved_target(cond)
        return False

    def get_active_tiles(self, state, action=[]):
        """
        The function `get_active_tiles` takes a state and an action as input and returns the indices of
        the active tiles based on the state variables and scaling factors.

        :param state: The state parameter is a list containing the values of the state variables. In
        this case, it contains four elements:
        :param action: The "action" parameter represents the action taken in the current state. It is
        used as an input to the "tiles" function to determine the active tiles
        :return: the indices of the active tiles.
        """
        # extract the state variables
        q1 = state[0]
        q2 = state[1]
        q1_d = state[2]
        q2_d = state[3]
        # define the scaling factor
        q1_sf = c.NUM_TILINGS * q1 / (c.WORKING_RANGE[0, 1] - c.WORKING_RANGE[0, 0])
        q2_sf = c.NUM_TILINGS * q2 / (c.WORKING_RANGE[1, 1] - c.WORKING_RANGE[1, 0])
        q1_d_sf = c.NUM_TILINGS * q1_d / (c.WORKING_VELOCITIES[0, 1] - c.WORKING_VELOCITIES[0, 0])
        q2_d_sf = c.NUM_TILINGS * q2_d / (c.WORKING_VELOCITIES[1, 1] - c.WORKING_VELOCITIES[1, 0])
        # get the indices of the active tiles
        active_tiles = tiles(self.iht, c.NUM_TILINGS, [q1_sf, q2_sf, q1_d_sf, q2_d_sf], action)
        return active_tiles

    def get_state_reward_action_transition(self, state, action, episode):
        """This function computes the next state and the reward for the current state and action.

        Args:
            state (np.array): current state
            action (np.array): current action

        Returns:
            np.array: next state
            float: reward
        """
        # compute the next state
        next_state = self.step(state, action)
        # compute the reward
        reward = self.get_reward(state, episode)
        return next_state, reward
