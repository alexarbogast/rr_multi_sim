import numpy as np

class Controller:
    def step(self, t, state, set_point, robot):
        return np.zeros(robot.n_joints)

class ComputedTorqueController(Controller):
    def __init__(self, Kp):
        self._Kp = 1

    def step(self, t, state, set_point, robot):
        s = set_point[:2]
        q, qd = state[:2], state[2:]

        u = self._Kp * (s - q)
        tau = robot._M(q)*(u) + robot._C(q, qd) + robot._G(q)
        return tau
