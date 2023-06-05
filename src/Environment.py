from math import tau
from operator import index
import constants as c
import numpy as np
from tiles3 import IHT, tiles


class Environ(object):
    def __init__(self, agent) -> None:
        self.__q = None
        self.__dq = None
        self.__reward = None
        self.__M = c.M  # number of tiles per dimension
        self.__N = c.N  # number of tiles to create
        self.__A = c.A  # number of actions
        self.agent = agent
        self.__alpha: float = c.ALPHA
        self.__gamma: float = c.GAMMA
        self.__sigma: float = c.SIGMA
        self.__epsilon: float = c.EPSILON
        self.__num_episodes = c.NUM_EPISODES
        self.__nCells = (self.__M**2) ** self.__A  # number of cells in the grid
        self.__d = self.__A * self.__N * self.__nCells  # number of weights
        self.__w = np.zeros(self.__d)  # weights
        self.__q1b, self.__q2b = c.WORKING_RANGE  # lower and upper bounds for q
        self.__dq1b, self.__dq2b = c.WORKING_VELOCITIES  # lower and upper bounds for dq
        self.__q0 = np.array([0, 0])  # initial state
        self.iht = IHT(self.__d)  # index hash table

    def get_info(self):
        print("Number of cells: ", self.__nCells)
        print("Number of weights: ", self.__d)
        print("Number of episodes: ", self.__num_episodes)
        print("Number of tiles per dimension: ", self.__M)
        print("Number of tiles to create: ", self.__N)
        print("Number of actions: ", self.__A)
        print("Bounds for q: ", self.__q1b, self.__q2b)
        print("Bounds for dq: ", self.__dq1b, self.__dq2b)
        print("Initial state: ", self.__q0)

    def build_tiles(self):
        """
        Builds the tiles for the grid.
        """
        offset = np.asarray(
            [1, 3]
        )  # offset for the grid according to the rule of the first odds numbers
        offset = offset / np.max(offset)  # normalize the offset

        # define the increment for the grid
        dq = (self.__ubq - self.__lbq) / self.__M
        q_span = [self.__lbq - q for q in np.arange(self.__lbq, self.__ubq, dq)]
        print(dq)

        ddq = (self.__ubdq - self.__lbdq) / self.__M
        dq_span = [self.__lbdq - dq for dq in np.arange(self.__lbdq, self.__ubdq, ddq)]

        # build the grid
        gridq = np.zeros((self.__N, len(q_span)))
        griddq = np.zeros((self.__N, len(dq_span)))

        # fill the first row of the grid
        gridq[0, :] = q_span
        griddq[0, :] = dq_span

        # fill the rest of the grid
        for i in range(1, self.__N):
            gridq[i, :] = dq_span + offset[0] * dq / self.__N * (i - 1)
            griddq[i, :] = dq_span + offset[1] * ddq / self.__N * (i - 1)

        return gridq, griddq

    def getRBF(self):
        """
        Returns the radial basis function approximation of a point in the plane.
        """

        # Define the output of the RBF
        phi = np.zeros((self.__d, 1))
        gridq, griddq = self.build_tiles()
        # get the action from the agent
        action = self.agent.get_action()
        # loop over the actions and the number of cells to fill the output
        for n in range(0, self.__N):
            for M1 in range(0, self.__M):
                for M2 in range(0, self.__M):
                    index = np.unravel_index(
                        indices=(M2, M1, n, action),
                        shape=(self.__M, self.__M, self.__N, self.__A),
                        order="F",
                    )
                    print(index)
                    phi[index] = np.exp(
                        -1
                        / self.__sigma**2
                        * np.linalg.norm(
                            self.agent.get_state() - [gridq[n, M1], griddq[n, M2]]
                        )
                        ** 2
                    )

        return phi

    def getFeatures(self):
        """
        Returns the features of the current state.
        """
        x = np.zeros((self.__d, 1))
        gridq, griddq = self.build_tiles()
        for n in range(self.__N):
            # get chunk of the grid
            q = gridq[n, :]
            dq = griddq[n, :]
            # get the indexes of the closest values that fall in the grid
            q_index = np.where(
                (
                    self.agent.get_state()[0] >= q[0:-2]
                    and self.agent.get_state()[0] < q[1:-1]
                )
                == True
            )[0][0]
            dq_index = np.where(
                (
                    self.agent.get_state()[1] >= dq[0:-2]
                    and self.agent.get_state()[1] < dq[1:-1]
                )
                == True
            )[0][0]
            # get the index of the action
            ind = np.unravel_index(
                shape=[self.__M, self.__M, self.__N, self.__A],
                indices=(q_index, dq_index, n, self.agent.get_action()),
                order="F",
            )
            # fill the features
            x[ind] = 1
        return x

    def epsGreedy(self):
        """
        Returns the action according to the epsilon greedy policy.
        The actions are chosen according to the weights of the RBF and represents
        the torque applied to the system, tau1 and tau2.
        """
        if np.random.rand() < self.__epsilon:
            # return a random action with probability epsilon
            return [np.random.randint(0, self.__A),
                    np.random.randint(0, self.__A)]
        else:
            tau1 = np.zeros((self.__A, 1))
            tau2 = np.zeros((self.__A, 1))
            for a in range(0, self.__A):
                tau1[a] = self.__w.T @ self.getRBF()
                tau2[a] = self.__w.T @ self.getRBF()
            a1 = np.unravel_index(np.argmax(tau1), tau1.shape)
            a2 = np.unravel_index(np.argmax(tau2), tau2.shape)
            return [a1, a2]

    def dynamics(self):
        """
        Returns the dynamics of the system integrated over a time step.
        x1 = q1
        x2 = q2
        x3 = dq1
        x4 = dq2
        dx3 = ddq1
        dx4 = ddq2
        """
        pass

    def evaluate_dynamics(self, x1, x2, x3, x4, tau1, tau2):
        dx3 = (
            self.agent.__L2
            * (
                2
                * self.agent.__L1
                * self.agent.__L2
                * self.agent.__M2
                * x3
                * x4**3
                * np.sin(x2)
                - self.agent.__L1
                * c.g
                * (self.agent.__M1 + self.agent.__M2)
                * np.cos(x1)
                - self.agent.__L2 * c.g * self.agent.__M2 * np.cos(x1 + x2)
                + tau1
            )
            + (self.agent.__L1 * np.cos(x2) + self.agent.__L2)
            * (
                self.agent.__L1
                * self.agent.__L2
                * self.agent.__M2
                * x3**2
                * np.sin(x2)
                + self.agent.__L2 * c.g * self.agent.__M2 * np.cos(x1 + x2)
                - tau2
            )
        ) / (
            self.agent.__L1**2
            * self.agent.__L2
            * (self.agent.__M1 + self.agent.__M2 * np.sin(x2) ** 2)
        )

        dx4 = (
            -self.agent.__L2
            * self.agent.__M2
            * (self.agent.__L1 * np.cos(x2) + self.agent.__L2)
            * (
                2
                * self.agent.__L1
                * self.agent.__L2
                * self.agent.__M2
                * x3
                * x4**3
                * np.sin(x2)
                - self.agent.__L1
                * c.g
                * (self.agent.__M1 + self.agent.__M2)
                * np.cos(x1)
                - self.agent.__L2 * c.g * self.agent.__M2 * np.cos(x1 + x2)
                + tau1
            )
            - (
                self.agent.__L1
                * self.agent.__L2
                * self.agent.__M2
                * x3**2
                * np.sin(x2)
                + self.agent.__L2 * c.g * self.agent.__M2 * np.cos(x1 + x2)
                - tau2
            )
            * (
                self.agent.__L1**2 * self.agent.__M1
                + self.agent.__L1**2 * self.agent.__M2
                + 2 * self.agent.__L1 * self.agent.__L2 * self.agent.__M2 * np.cos(x2)
                + self.agent.__L2**2 * self.agent.__M2
            )
        ) / (
            self.agent.__L1**2
            * self.agent.__L2**2
            * self.agent.__M2
            * (self.agent.__M1 + self.agent.__M2 * np.sin(x2) ** 2)
        )
        return np.array([dx3, dx4])

    def simulate(self, x0, u, dt):
        """
        Simulates the system for a time step dt.
        """
        x = np.zeros((4, 1))
        x[0] = x0[0]
        x[1] = x0[1]
        x[2] = x0[2]
        x[3] = x0[3]
        dx = np.zeros((4, 1))
        dx[0] = x0[2]
        dx[1] = x0[3]
        next_state = self.evaluate_dynamics(x[0], x[1], x[2], x[3], u[0], u[1])
        dx[2] = next_state[0]
        dx[3] = next_state[1]
        x = x + dx * dt
        return x

    def get_indexes(self, state, action):
        """
        Returns the tiles of the current state.
        """

        [q1, q2], [dq1, dq2] = state

        scaleFactor_q1 = self.__M / (self.__q1b[1] - self.__q1b[0])
        scaleFactor_q2 = self.__M / (self.__q2b[1] - self.__q2b[0])
        scaleFactor_dq1 = self.__M / (self.__dq1b[1] - self.__dq1b[0])
        scaleFactor_dq2 = self.__M / (self.__dq2b[1] - self.__dq2b[0])

        return tiles(
            self.iht,
            self.__M,
            [
                scaleFactor_q1 * q1,
                scaleFactor_dq1 * dq1,
                scaleFactor_q2 * q2,
                scaleFactor_dq2 * dq2,
            ],
            action,
        )
