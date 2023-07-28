from os import name
import numpy as np
from numpy import cos, sin
import constants as c


class Agent:
    def __init__(self, env=None):
        self.constants = {"l1": c.L1, "l2": c.L2, "m1": c.M1, "m2": c.M2, "g": c.g, "f1": c.f1, "f2": c.f2}
        self.is_sat = True
        self.achieved_target = False
        self.state = c.init_cond
        self.time_step = c.TIME_STEP
        self.actions: np.ndarray = [c.U1, c.U2] * np.asarray(
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
        self.env = env
        self.ws = self.generate_ws_curves()

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def set_env(self, env):
        self.env = env

    def get_env(self):
        return self.env

    def get_actions(self):
        return self.actions

    def get_selected_action(self, index):
        return self.actions[index]

    def get_random_action(self):
        return self.actions[np.random.randint(0, c.NUM_ACTIONS)]

    def get_achieved_target(self):
        return self.achieved_target

    def set_achieved_target(self, done):
        self.achieved_target = done

    def h_q(self, q1, q2):
        """Compute the forward kinematics of the robot.

        Args:
            q1: The angle of the first joint. Rad.
            q2: The angle of the second joint. Rad.

        Returns:
            A tuple containing the x and y coordinates of the end effector.
        """
        x1 = self.constants["l1"] * cos(q1)
        y1 = self.constants["l1"] * sin(q1)
        x2 = self.constants["l1"] * cos(q1) + self.constants["l2"] * cos(q1 + q2)
        y2 = self.constants["l1"] * sin(q1) + self.constants["l2"] * sin(q1 + q2)
        return np.asarray([x1, y1, x2, y2])

    def is_inside_ws(self, x, y):
        """Check if the current state is inside the workspace of the robot."""
        cond = ((x**2 + y**2) < (self.constants["l1"] + self.constants["l2"]) ** 2) & (
            (x**2 + y**2) > (self.constants["l1"] - self.constants["l2"]) ** 2
        )
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
        X, Y = self.h_q(grid[0], grid[1])[2:]
        # filter out the points that are outside the workspace
        mask = self.is_inside_ws(X, Y)
        X, Y = X[mask], Y[mask]
        # filter out duplicate of pairs of points
        indices = np.unique(np.stack((X, Y), axis=1), axis=0, return_index=True)[1]
        X, Y = X[indices], Y[indices]
        # return the points
        return X, Y

    def get_ws_curves(self):
        return self.ws

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
        f1 = self.constants["f1"]
        f2 = self.constants["f2"]
        # unpack the torques
        t1, t2 = tau
        # precompute some terms
        x3_sq = x3**2
        x4_sq = x4**2
        cos_x1 = cos(x1)
        cos_x2 = cos(x2)
        sin_x2 = sin(x2)
        cos_x1_x2 = cos_x1 * cos_x2 - sin(x1) * sin(x2)
        m2_l2_sq = l2**2 * m2
        m2_l1_l2_sq = l1 * m2_l2_sq
        m2_l1_l2_sq_2 = 2 * m2_l1_l2_sq
        m1_l1_l2_sq = m1 * l1 * m2_l2_sq / (m1 + m2)
        m2_l1_sq = m2 * l1**2
        m2_l1_sq_2 = 2 * m2_l1_sq
        m1_l1_sq = m1 * l1**2
        cos_x2_sq = cos_x2**2
        # compute the time derivative of the state vector
        dx1dt = x3
        dx2dt = x4
        dx3dt = (
            l2 * t1
            - l2 * t2
            - l1 * t2 * cos_x2
            - f1 * l2 * x3
            + f2 * l2 * x4
            + f2 * l1 * x4 * cos_x2
            + m2_l1_l2_sq * x3_sq * sin_x2
            + m2_l1_l2_sq * x4_sq * sin_x2
            + g * l1 * l2 * (m1 + m2) * cos_x1
            - g * m2_l1_l2_sq * cos_x1_x2 * cos_x2
            + m2_l1_sq * x3_sq * cos_x2 * sin_x2
            + m2_l1_l2_sq_2 * x3 * x4 * sin_x2
        ) / (m2_l1_sq * (l2 - m2_l2_sq * cos_x2_sq))

        dx4dt = (
            m2_l2_sq * t1
            - m1_l1_sq * t2
            - m2_l1_sq * t2
            - m2_l2_sq * t2
            - f1 * m2_l2_sq * x3
            + f2 * m1_l1_sq * x4
            + f2 * m2_l1_sq * x4
            + f2 * m2_l2_sq * x4
            + m2_l1_l2_sq * x3_sq * sin_x2
            + m1_l1_l2_sq * x3_sq * sin_x2
            + m2_l1_l2_sq * x4_sq * sin_x2
            - g * m2_l1_l2_sq * cos_x1_x2
            + g * m2_l2_sq * cos_x1
            + l1 * l2 * m2 * t1 * cos_x2
            - 2 * l1 * l2 * m2 * t2 * cos_x2
            + m1_l1_l2_sq * x3_sq * sin_x2
            + m2_l1_l2_sq_2 * x3 * x4 * sin_x2
            - g * m2_l2_sq * cos_x1_x2 * cos_x2
            - f1 * l1 * l2 * m2 * x3 * cos_x2
            + 2 * f2 * l1 * l2 * m2 * x4 * cos_x2
            + m2_l1_sq_2 * m2_l2_sq * x3_sq * cos_x2 * sin_x2
            + m2_l1_sq * m2_l2_sq * x4_sq * cos_x2 * sin_x2
            + g * m1_l1_sq * m2 * cos_x1 * cos_x2
            - g * m1_l1_l2_sq * m2 * cos_x1_x2
            + g * m2_l1_sq * m2 * cos_x1
            + m2_l1_sq_2 * m2_l2_sq * x3 * x4 * cos_x2 * sin_x2
            + g * m1_l1_l2_sq * m2 * cos_x1 * cos_x2
        ) / (-(m1_l1_l2_sq) * m2_l2_sq * cos_x2_sq + m2_l1_sq * m2_l2_sq + m1 * m2_l1_l2_sq)

        return np.asarray([dx1dt, dx2dt, dx3dt, dx4dt])

    # define the step function to compute the next state using RK4 integration
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
        next_state = self.get_state() + self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # saturate
        if self.is_sat:
            next_state[0] = np.clip(next_state[0], c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1])
            next_state[1] = np.clip(next_state[1], c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1])
            next_state[2] = np.clip(next_state[2], c.WORKING_VELOCITIES[0, 0], c.WORKING_VELOCITIES[0, 1])
            next_state[3] = np.clip(next_state[3], c.WORKING_VELOCITIES[1, 0], c.WORKING_VELOCITIES[1, 1])
        return next_state
