import re
import constants as c
import numpy as np
from numpy import ndarray, sin, cos
from numpy.linalg import norm


class agent(object):
    def __init__(self, target):
        self.constants = {"l1": c.L1, "l2": c.L2, "m1": c.M1, "m2": c.M2, "g": c.g}
        self.is_sat = True
        self.state = np.zeros((1, 4))
        self.time_step = c.TIME_STEP
        self.actions = 10000 * np.asarray(
            [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]
        )
        self.distance_history = np.zeros((c.NUM_EPISODES, 1))
        self.actions_history = [[]] * c.NUM_EPISODES
        self.reward_history = np.zeros((c.NUM_EPISODES, 1))
        self.position_history = np.zeros((int(c.MAX_STEPS), 2))
        self.target: ndarray = target

    def h_q(self, q1, q2):
        x = c.L1 * cos(q1) + c.L2 * cos(q1 + q2)
        y = c.L1 * sin(q1) + c.L2 * sin(q1 + q2)
        return np.asarray([x, y])

    def get_p(self):
        return self.h_q(self.state[0], self.state[1])

    def get_state(self):
        return self.state

    def get_action(self, index):
        return self.actions[index]

    def set_pos_history(self, episode):
        self.position_history[episode, :] = self.get_state()[0:2]

    def reset(self):
        # randomize initial joint angles
        self.state = np.asarray(
            [
                np.random.uniform(c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1]),
                np.random.uniform(c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1]),
                # velocities are set to zero when resetting
                0,
                0,
            ]
        )

    def is_inside_ws(self):
        """Check if the current state is inside the workspace of the robot."""
        x, y = self.get_p()
        cond = ((x**2 + y**2) < (c.L1 + c.L2) ** 2) & (
            (x**2 + y**2) > (c.L1 - c.L2) ** 2
        )
        return cond

    def compute_distance(self, state):
        """Compute the distance between the end effector and the target."""
        p = self.h_q(state[0], state[1])
        return norm(p - self.target)

    def get_reward(self, state, episode):
        """
        Returns the reward for the current state.
        """
        # compute the distance between the end effector and the target
        covered_distance = self.compute_distance(state)
        # compute the reward
        reward = -(covered_distance**2)
        # update the reward history
        self.reward_history[episode] += reward
        self.distance_history[episode] += covered_distance
        # return the reward
        return reward

    def is_done(self):
        """
        Checks if the current state is terminal.
        """
        p = self.h_q(self.state[0], self.state[1])
        if norm(p - self.target) <= c.EPSILON:
            return True
        return False

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
            2 * l2 * t1
            - 2 * l2 * t2
            - 2 * l2 * x3
            + 2 * l2 * x4
            - 2 * l1 * t2 * cos(x2)
            + 2 * l1 * x4 * cos(x2)
            - g * l1 * l2 * m2 * cos(x1 + 2 * x2)
            + l1**2 * l2 * m2 * x3**2 * sin(2 * x2)
            + 2 * l1 * l2**2 * m2 * x3**2 * sin(x2)
            + 2 * l1 * l2**2 * m2 * x4**2 * sin(x2)
            + 2 * g * l1 * l2 * m1 * cos(x1)
            + g * l1 * l2 * m2 * cos(x1)
            + 4 * l1 * l2**2 * m2 * x3 * x4 * sin(x2)
        ) / (l1**2 * l2 * (2 * m1 + m2 - m2 * cos(2 * x2)))
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
            + g * l1 * l2**2 * m2**2 * cos(x1)
            + l1**2 * l2**2 * m2**2 * x3**2 * sin(2 * x2)
            + (l1**2 * l2**2 * m2**2 * x4**2 * sin(2 * x2)) / 2
            + l1 * l2 * m2 * t1 * cos(x2)
            - 2 * l1 * l2 * m2 * t2 * cos(x2)
            - l1 * l2 * m2 * x3 * cos(x2)
            + 2 * l1 * l2 * m2 * x4 * cos(x2)
            + l1**3 * l2 * m1 * m2 * x3**2 * sin(x2)
            + 2 * l1 * l2**3 * m2**2 * x3 * x4 * sin(x2)
            + g * l1**2 * l2 * m2**2 * sin(x1) * sin(x2)
            + g * l1 * l2**2 * m1 * m2 * cos(x1)
            - g * l1 * l2**2 * m2**2 * cos(x1) * cos(x2) ** 2
            + l1**2 * l2**2 * m2**2 * x3 * x4 * sin(2 * x2)
            + g * l1 * l2**2 * m2**2 * cos(x2) * sin(x1) * sin(x2)
            + g * l1**2 * l2 * m1 * m2 * sin(x1) * sin(x2)
        ) / (l1**2 * l2**2 * m2 * (-m2 * cos(x2) ** 2 + m1 + m2))

        return np.asarray([dx1dt, dx2dt, dx3dt, dx4dt])

    # define the vector field function without friction
    def dxdt_no_friction(self, x, tau):
        """This function computes the time derivative of the state vector x without friction.

        Args:
            x (np.array): state vector
            constant (dictionary): constants dictionary for the system
            tau (list): torques

        Returns:
            np.array: time derivative of the state vector
        """
        # unpack the state vector
        x1, x2, x3, x4 = x[0]
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
            2 * l2 * t1
            - 2 * l2 * t2
            - 2 * l1 * t2 * cos(x2)
            - g * l1 * l2 * m2 * cos(x1 + 2 * x2)
            + l1**2 * l2 * m2 * x3**2 * sin(2 * x2)
            + 2 * l1 * l2**2 * m2 * x3**2 * sin(x2)
            + 2 * l1 * l2**2 * m2 * x4**2 * sin(x2)
            + 2 * g * l1 * l2 * m1 * cos(x1)
            + g * l1 * l2 * m2 * cos(x1)
            + 4 * l1 * l2**2 * m2 * x3 * x4 * sin(x2)
        ) / (l1**2 * l2 * (2 * m1 + m2 - m2 * cos(2 * x2)))
        dx4dt = -(
            l2**2 * m2 * t1
            - l1**2 * m2 * t2
            - l1**2 * m1 * t2
            - l2**2 * m2 * t2
            + l1 * l2**3 * m2**2 * x3**2 * sin(x2)
            + l1**3 * l2 * m2**2 * x3**2 * sin(x2)
            + l1 * l2**3 * m2**2 * x4**2 * sin(x2)
            + g * l1 * l2**2 * m2**2 * cos(x1)
            + l1**2 * l2**2 * m2**2 * x3**2 * sin(2 * x2)
            + (l1**2 * l2**2 * m2**2 * x4**2 * sin(2 * x2)) / 2
            + l1 * l2 * m2 * t1 * cos(x2)
            - 2 * l1 * l2 * m2 * t2 * cos(x2)
            + l1**3 * l2 * m1 * m2 * x3**2 * sin(x2)
            + 2 * l1 * l2**3 * m2**2 * x3 * x4 * sin(x2)
            + g * l1**2 * l2 * m2**2 * sin(x1) * sin(x2)
            + g * l1 * l2**2 * m1 * m2 * cos(x1)
            - g * l1 * l2**2 * m2**2 * cos(x1) * cos(x2) ** 2
            + l1**2 * l2**2 * m2**2 * x3 * x4 * sin(2 * x2)
            + g * l1 * l2**2 * m2**2 * cos(x2) * sin(x1) * sin(x2)
            + g * l1**2 * l2 * m1 * m2 * sin(x1) * sin(x2)
        ) / (l1**2 * l2**2 * m2 * (-m2 * cos(x2) ** 2 + m1 + m2))

        return np.asarray([dx1dt, dx2dt, dx3dt, dx4dt])

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
        self.state += self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # saturate
        if self.is_sat:
            # self.state[0] = np.clip(
            #     self.state[0], c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1]
            # )
            self.state[1] = np.clip(
                self.state[1], c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1]
            )
        return self.get_state()

    def step_no_friction(self, x, tau):
        """This function computes the next state of the system with RK4 method without friction.

        Args:
            x (np.array): state vector
            tau (list): torques

        Returns:
            np.array: next state of the system
        """
        # RK4 integration
        k1 = self.dxdt_no_friction(x, tau)
        k2 = self.dxdt_no_friction(x + self.time_step * k1 / 2, tau)
        k3 = self.dxdt_no_friction(x + self.time_step * k2 / 2, tau)
        k4 = self.dxdt_no_friction(x + self.time_step * k3, tau)
        # update the state
        self.state += self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # saturate
        if self.is_sat:
            self.state[0, 0] = np.clip(
                self.state[0, 0], c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1]
            )
            self.state[0, 1] = np.clip(
                self.state[0, 1], c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1]
            )
        return self.get_state()

    def get_state_reward_transition(self, state, action, episode):
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

    def get_state_reward_transition_no_friction(self, state, action):
        """This function computes the next state and the reward for the current state and action.

        Args:
            state (np.array): current state
            action (np.array): current action

        Returns:
            np.array: next state
            float: reward
        """
        # compute the next state
        next_state = self.step_no_friction(state, action)
        # compute the reward
        reward = self.get_reward(state)
        return next_state, reward
