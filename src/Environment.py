from operator import index
import constants as c
import numpy as np


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
            for m1 in range(0, self.__M):
                for m2 in range(0, self.__M):
                    index = np.unravel_index(
                        indices=(m2, m1, n, action),
                        shape=(self.__M, self.__M, self.__N, self.__A),
                        order="F",
                    )
                    print(index)
                    phi[index] = np.exp(
                        -1
                        / self.__sigma**2
                        * np.linalg.norm(
                            self.agent.get_state() - [gridq[n, m1], griddq[n, m2]]
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
        """
        if np.random.rand() < self.__epsilon:
            return np.random.randint(0, self.__A)
        else:
            q = np.zeros((self.__A, 1))
            for a in range(0, self.__A):
                q[a] = self.__w.T @ self.getRBF()
            a = np.unravel_index(np.argmax(q), q.shape)
            return a
