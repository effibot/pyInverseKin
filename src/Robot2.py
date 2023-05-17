import numpy as np
import pygame
import constants as c
from pygame.math import Vector2 as v
from pygame.gfxdraw import pixel as draw_pixel


# The class defines a robot with two links and joint angles, velocities, and end effector position, as
# well as a working range and center for PyGame rendering.
class Robot(object):
    def __init__(
        self,
        name: str = "Robot",  # robot's name
        l1: float = c.L1,  # length of the first link
        l2: float = c.L2,  # length of the second link
        working_range=c.WORKING_RANGE,  # working range of the robot
        center: np.ndarray = np.array([0, 0]),  # center of the robot in the screen
    ):
        self.robot_name: str = name
        self.__l1: float = l1
        self.__l2: float = l2
        self.__q: np.ndarray = np.zeros(2) + np.asarray(
            [
                -np.pi / 4,
                np.pi / 2,
            ]  # TODO> remove this, maybe set starting position from the environment
        )  # joint angles
        self.__dq: np.ndarray = np.zeros(2)  # joint velocities
        self.__p: np.ndarray = np.zeros(2)  # End effector position
        self.__J: np.ndarray = np.zeros((2, 2))  # Jacobian
        self.__working_range: np.ndarray = working_range
        self.__center: np.ndarray = center
        self.target = None  # target position
        # Generate the robot's link for PyGame Rendering
        self.__joints: list = self.update_joints()
        self.__action = np.array([-1, 0, 1])

    def forward_kinematics(self, q1, q2):
        """Compute the forward kinematics of the robot.

        Args:
            q1: The angle of the first joint. Rad.
            q2: The angle of the second joint. Rad.

        Returns:
            A tuple containing the x and y coordinates of the end effector.
        """
        x = self.__l1 * np.cos(q1) + self.__l2 * np.cos(q1 + q2)
        y = self.__l1 * np.sin(q1) + self.__l2 * np.sin(q1 + q2)
        return np.asarray((x, y))

    def inverse_kinematics(self, x, y, elbow_up=True):
        """Compute the inverse kinematics of the robot.

        Args:
            x: The x coordinate of the end effector. Meter.
            y: The y coordinate of the end effector. Meter.

        Returns:
            A tuple containing the angles of the two self.__joints.
        """
        # compute the angle of the second joint and find the two possible solutions
        c2 = (x**2 + y**2 - self.__l1**2 - self.__l2**2) / (
            2 * self.__l1 * self.__l2
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
        if (c2 == -1) | (self.__l1 == self.__l2):
            # if so, q1 is singular and we have infinite^1 solutions
            # in this case use the geometric solution
            q1 = np.clip(
                np.arctan2(y, x)
                - np.arctan2(
                    self.__l2 * np.sin(q2), self.__l1 + self.__l2 * np.cos(q2)
                ),
                -np.pi - 1e-6,  # we add a small offset to avoid numerical issues
                np.pi,
            )
        else:
            # use the algebraic solution - we skip the denominator since it is always positive
            c1_num = x * (self.__l1 + self.__l2 * np.cos(q2)) + y * self.__l2 * np.sin(
                q2
            )
            s1_num = y * (self.__l1 + self.__l2 * np.cos(q2)) - x * self.__l2 * np.sin(
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
        J[0, 0] = -self.__l1 * np.sin(q1) - self.__l2 * np.sin(q1 + q2)
        J[0, 1] = -self.__l2 * np.sin(q1 + q2)
        J[1, 0] = self.__l1 * np.cos(q1) + self.__l2 * np.cos(q1 + q2)
        J[1, 1] = self.__l2 * np.cos(q1 + q2)
        self.__J = J
        return J

    def __is_inside_ws(self, pose):
        """Check if the given pose is inside the workspace of the robot."""
        x, y = pose
        return ((x**2 + y**2) < (self.__l1 + self.__l2) ** 2) & (
            x**2 + y**2 > (self.__l1 - self.__l2) ** 2
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
        X, Y = X[self.__is_inside_ws((X, Y))], Y[self.__is_inside_ws((X, Y))]
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

    def getState(self):
        return self.__q

    def get_action(self):
        pass

    def update_joints(self):
        """
        The function updates the positions of the joints in a robotic arm based on the current joint
        angles. The positions of the joints are calculated using the forward kinematics equations,
        with specific offsets to account for the position of the base of the arm to get a correct rendering
        :return: a list of three tuples, which represent the positions of the three joints of a robotic
        arm. The positions are calculated based on the current values of the arm's joint angles and
        lengths.
        """
        return list(
            map(
                v,
                [
                    (self.__center[0], self.__center[1]),
                    (
                        self.__center[0] + self.__l1 * np.cos(self.__q[0]),
                        self.__center[1] - self.__l1 * np.sin(self.__q[0]),
                    ),
                    (
                        self.__center[0]
                        + self.__l1 * np.cos(self.__q[0])
                        + self.__l2 * np.cos(self.__q[0] + self.__q[1]),
                        self.__center[1]
                        - self.__l1 * np.sin(self.__q[0])
                        - self.__l2 * np.sin(self.__q[0] + self.__q[1]),
                    ),
                ],
            )
        )
