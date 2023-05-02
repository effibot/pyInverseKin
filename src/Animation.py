"""
Module to generate animation of the end-effector trajectory.
"""
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


class Animation(object):
    def __init__(self, robot, trajectory):
        self.plt = plt
        self.robot = robot
        self.trajectory = trajectory
        self.__animation_dMat = np.zeros((1, 4), dtype=np.float32)

    def __animation_data_generation(self):
        """
        Description:
            Generation data for animation. The resulting matrix {self.__animation_dMat} will contain each of the positions for the animation objects.
        """

        self.__animation_dMat = np.zeros(
            (len(self.trajectory.trajectory[0]), 4), dtype=np.float32
        )

        for i in range(len(self.trajectory.trajectory[0])):
            self.robot.inverse_kinematics(
                [self.trajectory.trajectory[0][i], self.trajectory.trajectory[1][i]],
                self.trajectory.trajectory[2][i],
            )
            self.__animation_dMat[i][0] = self.robot.rDH_param.a[0] * np.cos(
                self.robot.rDH_param.theta[0]
            )
            self.__animation_dMat[i][1] = self.robot.rDH_param.a[0] * np.sin(
                self.robot.rDH_param.theta[0]
            )
            self.__animation_dMat[i][2] = self.robot.pose[0]
            self.__animation_dMat[i][3] = self.robot.pose[1]

    def init_animation(self):
        """
        Description:
            Initialize each of the animated objects that will move in the animation.

        Returns:
            (1) parameter{1} [Float Array]: Arrays of individual objects
        """

        # Generation data for animation
        self.__animation_data_generation()

        # Initialization of robot objects
        self.robot.__line[0].set_data(
            [0.0, self.__animation_dMat[0][0]], [0.0, self.__animation_dMat[0][1]]
        )
        self.robot.__line[1].set_data(
            [self.__animation_dMat[0][0], self.__animation_dMat[0][2]],
            [self.__animation_dMat[0][1], self.__animation_dMat[0][3]],
        )
        self.robot.__line[2].set_data(
            self.__animation_dMat[0][0], self.__animation_dMat[0][1]
        )
        self.robot.__line[3].set_data(
            self.__animation_dMat[0][2], self.__animation_dMat[0][3]
        )

        return self.robot.__line
