import numpy as np
import pygame
import constants as c
from pygame.math import Vector2 as v
from pygame.gfxdraw import pixel as draw_pixel
from simulation import Dynamic


# The class defines a robot with two links and joint angles, velocities, and end effector position, as
# well as a working range and center for PyGame rendering.
class Robot(object):
    def __init__(
        self,
        name: str = "Robot",  # robot's name
        l1: float = c.L1,  # length of the first link
        l2: float = c.L2,  # length of the second link
        m1: float = c.M1,  # mass of the first link
        m2: float = c.M2,  # mass of the second link
        working_range=c.WORKING_RANGE,  # working range of the robot
        center: np.ndarray = np.array([0, 0]),  # center of the robot in the screen
    ):
        self.robot_name: str = name
        self.__L1: float = l1
        self.__L2: float = l2
        self.__M1: float = m1
        self.__M2: float = m2
        self.__q: np.ndarray = c.init_cond[0]  # joint angles
        self.__dq: np.ndarray = c.init_cond[1]  # joint velocities
        self.__p: np.ndarray = c.init_cond[2]  # End effector position
        self.__J: np.ndarray = np.zeros((2, 2))  # Jacobian
        self.__working_range: np.ndarray = working_range
        self.__center: np.ndarray = center
        self.target: np.ndarray = np.zeros((2, 2))  # target position
        # Generate the robot's link for PyGame Rendering
        self.__joints: list = self.update_joints()
        self.__actions = np.array(
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
        )  # action to be taken
        self.__distance_history = np.zeros((c.NUM_EPISODES, 1))
        self.__system = Dynamic()

    def forward_kinematics(self, q1, q2):
        """Compute the forward kinematics of the robot.

        Args:
            q1: The angle of the first joint. Rad.
            q2: The angle of the second joint. Rad.

        Returns:
            A tuple containing the x and y coordinates of the end effector.
        """
        x = self.__L1 * np.cos(q1) + self.__L2 * np.cos(q1 + q2)
        y = self.__L1 * np.sin(q1) + self.__L2 * np.sin(q1 + q2)
        return np.asarray((x, y))

    def compute_fwd_kin(self):
        """Compute the forward kinematics of the robot."""
        return self.forward_kinematics(self.__q[0], self.__q[1])

    def inverse_kinematics(self, x, y, elbow_up=True):
        """Compute the inverse kinematics of the robot.

        Args:
            x: The x coordinate of the end effector. Meter.
            y: The y coordinate of the end effector. Meter.

        Returns:
            A tuple containing the angles of the two self.__joints.
        """
        # compute the angle of the second joint and find the two possible solutions
        c2 = (x**2 + y**2 - self.__L1**2 - self.__L2**2) / (
            2 * self.__L1 * self.__L2
        )
        # find the two possible solutions for the second joint
        if c2 > 1 or c2 < -1:
            # no feasible solution, fit the elbow to the nearest feasible position
            c2 = np.clip(c2, -1, 1)
            q2 = np.arctan2(np.sqrt(1 - c2**2), c2)
        else:
            # feasible solution, compute the two possible solutions for selected elbo
            if elbow_up:
                q2 = np.arctan2(np.sqrt(1 - c2**2), c2)
            else:
                q2 = np.arctan2(-np.sqrt(1 - c2**2), c2)
        # compute the angle of the first joint
        if (c2 == -1) | (self.__L1 == self.__L2):
            # if so, q1 is singular and we have infinite^1 solutions
            # in this case use the geometric solution
            q1 = np.clip(
                np.arctan2(y, x)
                - np.arctan2(
                    self.__L2 * np.sin(q2), self.__L1 + self.__L2 * np.cos(q2)
                ),
                -np.pi - 1e-6,  # we add a small offset to avoid numerical issues
                np.pi,
            )
        else:
            # use the algebraic solution - we skip the denominator since it is always positive
            c1_num = x * (self.__L1 + self.__L2 * np.cos(q2)) + y * self.__L2 * np.sin(
                q2
            )
            s1_num = y * (self.__L1 + self.__L2 * np.cos(q2)) - x * self.__L2 * np.sin(
                q2
            )
            q1 = np.arctan2(s1_num, c1_num)
        return np.asarray((q1, q2))

    def jacobian(self, q1, q2):
        """Compute the Jacobian matrix of the robot.

        Args:
            q1: The angle of the first joint.
            q2: The angle of the second joint.

        Returns:
            The Jacobian matrix of the robot.
        """
        J = np.zeros((2, 2))
        J[0, 0] = -self.__L1 * np.sin(q1) - self.__L2 * np.sin(q1 + q2)
        J[0, 1] = -self.__L2 * np.sin(q1 + q2)
        J[1, 0] = self.__L1 * np.cos(q1) + self.__L2 * np.cos(q1 + q2)
        J[1, 1] = self.__L2 * np.cos(q1 + q2)
        self.__J = J
        return J

    def _is_inside_ws(self, pose):
        """Check if the given pose is inside the workspace of the robot."""
        x, y = pose
        return ((x**2 + y**2) < (self.__L1 + self.__L2) ** 2) & (
            x**2 + y**2 > (self.__L1 - self.__L2) ** 2
        )

    def generate_ws_curves(self, n):
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
        t1 = np.linspace(
            self.__working_range[0][0], self.__working_range[0][1], n_points
        )[:-1]
        t2 = np.linspace(
            self.__working_range[1][0], self.__working_range[1][1], n_points
        )[:-1]
        # create the grid of points
        grid = np.meshgrid(t1, t2)
        # compute the forward kinematics for each point in the grid
        X, Y = self.forward_kinematics(grid[0], grid[1])
        # filter out the points that are outside the workspace
        X, Y = X[self._is_inside_ws((X, Y))], Y[self._is_inside_ws((X, Y))]
        # filter out duplicate of pairs of points
        X, Y = np.unique([X, Y], axis=1)
        # return the points
        return X, Y

    def render(self, screen: pygame.Surface, ws: bool = False):
        """
        This function renders a robotic arm on a Pygame surface, with the option to also draw a
        workspace.

        :param screen: `screen` is a pygame.Surface object representing the display surface on which the
        robot arm will be rendered
        :type screen: pygame.Surface
        :param ws: The "ws" parameter is a boolean flag that determines whether or not to draw the
        workspace of the robotic arm. If it is set to True, the workspace will be drawn on the screen.
        If it is set to False, the workspace will not be drawn, defaults to False
        :type ws: bool (optional)
        """
        # draw workspace
        if ws:
            X, Y = self.generate_ws_curves(100)
            X, Y = X + self.__center[0], Y + self.__center[1]
            for i in range(len(X) - 1):
                draw_pixel(screen, int(X[i]), int(Y[i]), c.GREEN)
        # draw base
        pygame.draw.rect(
            screen,
            c.GREY,
            pygame.Rect(self.__center[0] - 10, self.__center[1] - 10, 20, 20),
        )
        # draw links
        for i in range(1, len(self.__joints)):
            prev = self.__joints[i - 1]
            curr = self.__joints[i]
            rad = 7
            if i == 1:
                # first joint
                color = c.BLUE
            else:
                # other joints
                color = c.GREY
            # draw link
            L = c.L1 if i == 1 else c.L2
            # pygame.draw.rect(screen, c.BLACK, [prev[0], prev[1] - 5, L, 10])
            pygame.draw.aaline(screen, c.BLACK, prev, curr, 1)
            # draw joint
            if i == 1:
                pygame.draw.circle(screen, c.GREY, curr, radius=rad + 2)
            pygame.draw.circle(screen, color, prev, radius=rad)
            if i == len(self.__joints) - 1:
                # draw end effector
                pygame.draw.circle(screen, c.GREY, curr, radius=rad + 2)
                pygame.draw.circle(screen, c.GREEN, curr, radius=rad)

    def get_p(self):
        return self.__p

    def get_q(self):
        return self.__q

    def get_dq(self):
        return self.__dq

    def get_action(self, index):
        return self.__actions[index]

    def get_state(self):
        """
        Return the current state as a list of [q1, dq1, q2, dq2]
        """
        return [self.__q[0], self.__dq[0], self.__q[1], self.__dq[1]]

    def update_joints(self):
        """
        The function updates the positions of the joints in a robotic arm based on the current joint
        angles. The positions of the joints are calculated using the forward kinematics equations,
        with specific offsets to account for the position of the base of the arm to get a correct rendering
        :return: a list of three tuples, which represent the positions of the three joints of a robotic
        arm. The positions are calculated based on the current values of the arm's joint angles and
        lengths.
        """
        new_list = list(
            map(
                v,
                [
                    (self.__center[0], self.__center[1]),
                    (
                        self.__center[0] + self.__L1 * np.cos(self.__q[0]),
                        self.__center[1] - self.__L1 * np.sin(self.__q[0]),
                    ),
                    (
                        self.__center[0]
                        + self.__L1 * np.cos(self.__q[0])
                        + self.__L2 * np.cos(self.__q[0] + self.__q[1]),
                        self.__center[1]
                        - self.__L1 * np.sin(self.__q[0])
                        - self.__L2 * np.sin(self.__q[0] + self.__q[1]),
                    ),
                ],
            )
        )
        self.__joints = new_list
        return self.__joints

    def set_joint_angles(self, q):
        self.__q = q
        self.update_joints()

    def reset(self):
        # randomize initial joint angles
        self.__q = np.array(
            [
                np.random.uniform(
                    self.__working_range[0][0], self.__working_range[0][1]
                ),
                np.random.uniform(
                    self.__working_range[1][0], self.__working_range[1][1]
                ),
            ]
        )
        # velocities are set to zero when resetting
        self.__dq = np.array([0, 0])

    def get_reward(self, episode):
        """
        Returns the reward for the current state.
        """
        # calculate the distance between the end effector and the target

        if self._is_inside_ws(self.__p):
            covered_distance = np.linalg.norm(self.__p - self.target) ** 2
            self.__distance_history[episode] = covered_distance
            return -covered_distance
        return -100

    def get_next_state(self, state, action, t):
        """
        This function calculates the next state of the robotic arm based on the current state and
        the action taken. The next state is calculated using the forward kinematics equations, with
        the addition of the action taken to the current joint angles.

        :param action: The action taken by the  The action is a list of two values, which
        represent the increments to be added to the current joint angles.
        :type action: list
        :return: a list of four values, which represent the angles and angular velocities
        of the two joints of the robotic arm at the next state.
        """
        # unpack the elapsed time
        ti, tf = t
        # assign the torque values to variables
        tau1, tau2 = action
        # from the current state, calculate the next state
        next_state = self.__system.step(
            state, [ti, tf], tau1, tau2, self.__M1, self.__M2, c.g, self.__L1, self.__L2
        )
        # update the joint angles
        self.set_q(np.array([next_state[0], next_state[2]]))
        # update the joint velocities
        self.set_dq(np.array([next_state[1], next_state[3]]))
        #print(f"state: {state}\t action: {action}\t next_state: {next_state}")
        self.update_joints()
        return self.get_state()

    def set_q(self, q):
        self.__q = q

    def set_dq(self, dq):
        self.__dq = dq

    def set_target(self, target):
        """
        Sets the target for the robotic arm.
        """
        self.target = target
