import constants as c
import numpy as np


class Environ(object):
    def __init__(self, agent) -> None:
        self.__q = None
        self.__dq = None
        self.__reward = None
        self.__sigma = None
        self.__M = c.M  # number of radial basis functions
        self.__N = c.N
        self.__A = c.A  # number of actions
        self.agent = agent
        self.__alpha = c.ALPHA
        self.__gamma = c.GAMMA
        self.__sigma = c.SIGMA
        self.__num_episodes = c.NUM_EPISODES
        self.__nCells = (self.__M**2) ** self.__A  # number of cells in the grid
        self.__d = self.__A * self.__N * self.__nCells  # number of weights
        self.__w = np.zeros(self.__d)  # weights
        (self.__lbq, self.__ubq), (
            self.__lbdq,
            self.__ubdq,
        ) = c.WORKING_VELOCITIES  # lower and upper bounds for q and dq
        pass

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

    def getRBF(self, s, a, sigma, gridq, griddq, M, N, A):
        """
        Returns the radial basis function approximation of a point in the plane.
        """
        pass
