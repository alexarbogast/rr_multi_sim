import numpy as np

class Robot2R:
    def __init__(self, m1, m2, l1, l2, g=9.81, base=(0,0)):
        self._m1 = m1
        self._m2 = m2
        self._l1 = l1
        self._l2 = l2
        self._g  = g
        self._base = base

        self.n_joints = 2

    def _M(self, q):
        """ Mass matrix """
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
        """ Coriolis-centrifugal matrix """
        m2 = self._m2
        l1l2 = self._l1*self._l2

        C1 = -m2*l1l2*np.sin(q[1])*(2*qd[0]*qd[1] + qd[1]*qd[1])
        C2 =  m2*l1l2*qd[0]*qd[0]*np.sin(q[1])

        return np.array([[C1], [C2]])

    def _G(self, q):
        """ Gravity matrix """
        m1, m2 = self._m1, self._m2
        l1, l2 = self._l1, self._l2
        g = self._g

        G1 = (m1 + m2)*l1*g*np.cos(q[0]) + m2*g*l2*np.cos(q[0] + q[1])
        G2 = m2*g*l2*np.cos(q[0] + q[1])

        return np.array([[G1], [G2]])

    def forward_dynamics(self, q, qd, tau=0):
        """ 
        Forward dynamics for RR manipulator
        
        M(q)*qdd + C(q, qd) + G(q) = tau

        Parameters
        ----------
        q : numpy.array[float]
            robot joint angles
        qd : numpy.array[float]
            robot joint velocities
        tau : numpy.array[float]
            input torque

        Returns
        -------
        qdd : numpy.array[float]
            robot joint accelerations 
        """

        qdd = np.linalg.inv(self._M(q)) @ (tau - self._C(q, qd) - self._G(q))
        return qdd

    def forward_kinematics(self, q):
        """
        Forward kinematics for RR manipulator

        Parameters
        ----------
        q : numpy.array[float]
            robot joint angles

        Returns
        -------
        numpy.ndarray[float]
            frames [p0, p1, p2] in world frame
        """

        l1, l2 = self._l1, self._l2

        p0 = np.array(self._base)
        p1 = np.array([l1*np.cos(q[0]), l1*np.sin(q[0])]) + p0
        p2 = p1 + np.array([l2*np.cos(q[0] + q[1]), l2*np.sin(q[0] + q[1])])
        return np.vstack((p0,p1,p2))

    def inverse_kinematics(self, pos, elbow_up=True):
        """
        Inverse kinematics for RR manipulator

        Parameters
        ----------
        pos : np.array[float]
            [xpos ypos]
        elbow_up : bool
            configuration choice

        Returns
        -------
        sol : np.array[float]
            [q1, q2]
        """

        l1l1 = self._l1*self._l1
        l2l2 = self._l2*self._l2
        l1l2 = self._l1*self._l2

        x = pos[0] - self._base[0]
        y = pos[1] - self._base[1]

        B = np.arccos((l1l1 + l2l2 - x*x - y*y)/(2*l1l2))
        A = np.arccos((x*x + y*y + l1l1 - l2l2)/(2*self._l1*np.sqrt(x*x + y*y)))
        S = np.arctan2(y, x)

        sol = None
        if elbow_up:
            sol = np.array([S + A, B - np.pi]) 
        else:
            sol = np.array([S - A, np.pi - B])
        return sol

    def jacobian(self, q):
        """ Jacobian for RR manipulator """
        l1, l2 = self._l1, self._l2
        
        J = np.array([[-l1*np.sin(q[0])-l2*np.sin(q[0]+q[1]), -l2*np.sin(q[0]+q[1])],
                      [ l1*np.cos(q[0])+l2*np.cos(q[0]+q[1]),  l2*np.cos(q[0]+q[1])]])
        return J


class Positioner:
    # 1-DOF Rotary Table
    def __init__(self, m, r):
        self._m = m
        self._r = r
        self._I = 0.5*m*r*r

    def forward_dynamics(self, q, qd, tau=0):
        # Iz * qdd = tau
        qdd = 1/self._I * tau
        return qdd