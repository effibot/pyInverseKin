import constants as c
import numpy as np
import sympy as sp
from scipy.integrate import odeint
from sympy import cos, sin

# Define global symbols for the equations
M1, M2, L1, L2, g, t, tau1, tau2 = sp.symbols("M1 M2 L1 L2 g t tau1 tau2")
x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", cls=sp.Function)
x = sp.symbols("x")


# The class "Dynamic" defines a system of differential equations for a 2 revolute link robot arm
# and provides methods to numerically solve them.
class Dynamic:
    def __init__(self, N=1000):
        self.num_points = N
        self.__x1, self.__x2, self.__x3, self.__x4 = x1(t), x2(t), x3(t), x4(t)
        self.__x3_d = (
            L2
            * (
                2 * L1 * L2 * M2 * x3(t) * x4(t) ** 3 * sin(self.__x2)
                - L1 * g * (M1 + M2) * cos(self.__x1)
                - L2 * g * M2 * cos(self.__x1 + self.__x2)
                - tau1
            )
            + (L1 * cos(self.__x2) + L2)
            * (
                L1 * L2 * M2 * x3(t) ** 2 * sin(self.__x2)
                + L2 * g * M2 * cos(self.__x1 + self.__x2)
                + tau2
            )
        ) / (L1**2 * L2 * (M1 + M2 * sin(self.__x2) ** 2))
        self.__x4_d = (
            L2
            * M2
            * (L1 * cos(self.__x2) + L2)
            * (
                -2 * L1 * L2 * M2 * x3(t) * x4(t) ** 3 * sin(self.__x2)
                + L1 * g * (M1 + M2) * cos(self.__x1)
                + L2 * g * M2 * cos(self.__x1 + self.__x2)
                + tau1
            )
            - (
                L1 * L2 * M2 * x3(t) ** 2 * sin(self.__x2)
                + L2 * g * M2 * cos(self.__x1 + self.__x2)
                + tau2
            )
            * (
                L1**2 * M1
                + L1**2 * M2
                + 2 * L1 * L2 * M2 * cos(self.__x2)
                + L2**2 * M2
            )
        ) / (L1**2 * L2**2 * M2 * (M1 + M2 * sin(self.__x2) ** 2))
        self.x1_d_f = sp.lambdify(self.__x1.diff(t), self.__x1.diff(t))
        self.x2_d_f = sp.lambdify(self.__x2.diff(t), self.__x2.diff(t))
        self.x3_d_f = sp.lambdify(
            [
                t,
                self.__x1,
                self.__x2,
                self.__x3,
                self.__x4,
                tau1,
                tau2,
                M1,
                M2,
                g,
                L1,
                L2,
            ],
            self.__x3_d,
        )
        self.x4_d_f = sp.lambdify(
            [
                t,
                self.__x1,
                self.__x2,
                self.__x3,
                self.__x4,
                tau1,
                tau2,
                M1,
                M2,
                g,
                L1,
                L2,
            ],
            self.__x4_d,
        )
        x1_sat = sp.lambdify(
            args=x,
            expr=sp.Piecewise((-np.pi, x <= -np.pi), (np.pi, x >= np.pi), (x, True)),
            modules="numpy",
        )
        x2_sat = sp.lambdify(
            args=x,
            expr=sp.Piecewise(
                (-3 / 4 * np.pi, x <= -3 / 4 * np.pi),
                (3 / 4 * np.pi, x >= 3 / 4 * np.pi),
                (x, True),
            ),
            modules="numpy",
        )
        x34_sat = sp.lambdify(
            args=x,
            expr=sp.Piecewise(
                (-np.pi / 180, x <= np.pi / 180),
                (np.pi / 180, x >= np.pi / 180),
                (x, True),
            ),
            modules="numpy",
        )
        self.sym_res = np.zeros((self.num_points, 4))

    def __dxdt(self, x, t, tau1, tau2, M1, M2, g, L1, L2):
        """
        The function calculates the derivatives of the state variables.

        :param x: a list of the current values of the state variables x1, x2, x3, and x4. x1 and x2 are the angles of the first and second joints, respectively, and x3 and x4 are the angular velocities of the first and second joints, respectively.
        :param t: time
        :param tau1: tau1 is a parameter representing the torque applied to the first joint.
        :param tau2: tau2 is a parameter representing the torque applied to the second joint.
        :param M1: mass of the first link
        :param M2: mass of the second link
        :param g: acceleration due to gravity
        :param L1: The length of the first link
        :param L2: The length of the second link
        :return: The function `dxdt` returns a list of four elements, which are the values of the
        derivatives of the state variables `x1`, `x2`, `x3`, and `x4` at a given time `t`, based on the
        current values of the state variables and the input parameters `tau1`, `tau2`, `M1`, `M2`, `g`,
        """
        # unpack the state variables
        x1, x2, x3, x4 = x
        # define the derivatives of the state variables as a list of four elements according to
        # the odeint function's requirements wich defines the vector field of the system as
        # S(q) = [q1, q1', q2, q2'] => S(x) = [x1, x3, x2, x4]
        return [
            self.x1_d_f(x3),  # x1_d = q1_d
            self.x3_d_f(
                t, x1, x2, x3, x4, tau1, tau2, M1, M2, g, L1, L2
            ),  # x3_d = q1_dd
            self.x2_d_f(x4),  # x2_d = q2_d
            self.x4_d_f(
                t, x1, x2, x3, x4, tau1, tau2, M1, M2, g, L1, L2
            ),  # x4_d = q2_dd
        ]  # S(q') = [q1', q1'', q2', q2''] => S(x') = [x3, x3_d, x4, x4_d]

    def simulate(self, x0, t, tau1, tau2, M1, M2, g, L1, L2):
        """
        This function simulates a system using the given initial conditions and parameters by
        numerically solving a set of differential equations using the odeint function.
        """
        ti, tf = t
        time = np.linspace(ti, tf, self.num_points, endpoint=True)
        self.sym_res[0] = x0
        for i in range(1, self.num_points):
            x = odeint(
                self.__dxdt,
                self.sym_res[i - 1],
                [time[i - 1], time[i]],
                args=(tau1, tau2, M1, M2, g, L1, L2),
            )
            self.sym_res[i] = x[-1]

    def step(self, x0, t, tau1, tau2, M1, M2, g, L1, L2):
        """
        The function calculates the final state of a system using the given initial state and parameters
        by solving a system of ordinary differential equations using the odeint function.

        :return: the final state of the system as a list of [x1, x2, x3, x4] = [q1, q1_d, q2, q2_d]
        """
        ti, tf = t
        x = odeint(
            self.__dxdt,
            x0,
            [ti, tf],
            args=(tau1, tau2, M1, M2, g, L1, L2),
        )
        return x[-1]


if __name__ == "__main__":
    g_ = 9.81
    m1 = 2
    m2 = 1
    L1_ = 2
    L2_ = 1
    x0 = [-3.0140259566186343, 0.0, 0.5003789423334926, 0.0]
    num_points = 1001
    ti = 0
    tf = 0.01
    dyna = Dynamic(num_points)
    simulate = dyna.step(x0, [ti, tf], 0, 0, m1, m2, g_, L1_, L2_)
    print(simulate)
