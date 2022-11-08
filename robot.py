import numpy as np

class Robot2R:
    def __init__(self, m1, m2, l1, l2, g=9.81):
        self._m1 = m1
        self._m2 = m2
        self._l1 = l1
        self._l2 = l2
        self._g  = g

    def _M(self, q):
        m1, m2 = self._m1, self._m2

        l1l1 = self._l1*self._l1
        l2l2 = self._l2*self._l2
        l1l2 = self._l1*self._l2

        M11 = m1*l1l1 + m2*(l1l1 + 2*l1l2*np.cos(q[1]) + l2l2)
        M12 = m2*(l1l2*np.cos(q[1]) + l2l2)
        M21 = m2*(l1l2*np.cos(q[1]) + l2l2)
        M22 = m2*l2l2

        return np.array([[M11, M12], [M21, M22]])

    def _C(self, q, qd):
        m2 = self._m2
        l1l2 = self._l1*self._l2

        C1 = -m2*l1l2*np.sin(q[1])*(2*qd[0]*qd[1] + qd[1]*qd[1])
        C2 =  m2*l1l2*qd[0]*qd[0]*np.sin(q[1])

        return np.array([[C1], [C2]])

    def _G(self, q):
        m1, m2 = self._m1, self._m2
        l1, l2 = self._l1, self._l2
        g = self._g

        G1 = (m1 + m2)*l1*g*np.cos(q[0]) + m2*g*l2*np.cos(q[0] + q[1])
        G2 = m2*g*l2*np.cos(q[0] + q[1])

        return np.array([[G1], [G2]])

    def forward_dynamics(self, q, qd, tau=0):
        qdd = np.linalg.inv(self._M(q)) @ (tau - self._C(q, qd) - self._G(q))
        return qdd

    def forward_kinematics(self, q):
        l1, l2 = self._l1, self._l2

        p0 = np.array([0, 0])
        p1 = np.array([l1*np.cos(q[0]), l1*np.sin(q[0])])
        p2 = p1 + np.array([l2*np.cos(q[0] + q[1]), l2*np.sin(q[0] + q[1])])
        return np.vstack((p0,p1,p2))