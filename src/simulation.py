import constants as c
import numpy as np
from numpy import sin, cos
import sympy as sp
from scipy.integrate import odeint

# Define global symbols for the equations
M1, M2, L1, L2, g, t, tau1, tau2 = sp.symbols("M1 M2 L1 L2 g t tau1 tau2")
x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", cls=sp.Function)
x = sp.symbols("x")


# The class "Dynamic" defines a system of differential equations for a 2 revolute link robot arm
# and provides methods to numerically solve them.
class system:
    def __init__(self, N=1000):
        self.num_points = N
        self.constants = {"l1": c.L1, "l2": c.L2, "m1": c.M1, "m2": c.M2, "g": c.g}
        self.is_sat = True
        self.state = np.zeros((1, 4))
        self.time_step = c.TIME_STEP

    def get_state(self):
        return self.state

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
            self.state[0, 0] = np.mod(self.state[0, 0] + np.pi, 2 * np.pi) - np.pi
            self.state[0, 1] = (
                np.mod(self.state[0, 1] + 5 / 6 * np.pi, 10 / 6 * np.pi) - 5 / 6 * np.pi
            )
        return self.get_state()
