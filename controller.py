import numpy as np

class Controller:
    def step(self, t, state, set_point, robot):
        return np.zeros(robot.n_joints)

class ComputedTorqueController(Controller):
    def __init__(self, Kp, Kd):
        self._Kp = Kp
        self._Kd = Kd

    def step(self, t, state, set_point, robot):
        """
        Find control signal for step at time t.
        
        Parameters
        ----------
        t : float
            time value
        state : numpy.ndarray[float]
            robot2R state = [q1, q2, q1d, q2d]
        set_point : numpy.ndarray[float]
            desired position and velocity = [q1, q2, q1d, q2d]
        robot : Robot2R
            a 2R robot object

        Returns
        -------
        tau : float
            torque for robot joints
        """

        s, sd = set_point[:2], set_point[2:4]
        q, qd = state[:2], state[2:]

        e = s - q
        ed = sd - qd
        u = self._Kp @ e + self._Kd @ ed

        tau = robot._M(q) @ u[:, np.newaxis] + robot._C(q, qd) + robot._G(q)
        return tau


class InverseJacobianController(Controller):
    def __init__(self, Kp):
        self._Kp = Kp 

    def step(self, t, state, set_point, robot):
        """
        Find control signal for step at time t.

        v = J(q)*u
        u = Kp*ep + [xd_d; yd_d]
        
        Parameters
        ----------
        t : float
            time value
        state : numpy.ndarray[float]
            robot2R state = [q1, q2, q1d, q2d]
        set_point : numpy.ndarray[float]
            desired position and velocity = [x_d, y_d, xd_d, yd_d]
        robot : Robot2R
            a 2R robot object

        Returns
        -------
        qd : numpy.ndarray[float]
            joint velocity
        """

        *_, p = robot.forward_kinematics(state[:2])
        ep = set_point[:2] - p
        u = (self._Kp @ ep) + set_point[2:]

        J = robot.jacobian(state)
        Jinv = np.linalg.inv(J)
        qd = Jinv @ u
        return qd

    
class TrajectoryController(Controller):
    def __init__(self, Kp, Kd, Kp_jac):
        self._ctc = ComputedTorqueController(Kp, Kd)
        self._ijc = InverseJacobianController(Kp_jac)

    def step(self, t, state, set_point, robot):
        """
        Find control signal for step at time t.
        
        Parameters
        ----------
        t : float
            time value
        state : numpy.ndarray[float]
            robot2R state = [q1, q2, q1d, q2d]
        set_point : numpy.ndarray[float]
            desired position and velocity = [x_d, y_d, xd_d, yd_d]
        robot : Robot2R
            a 2R robot object

        Returns
        -------
        tau : float
            torque for robot joints
        """

        qd = self._ijc.step(t, state, set_point, robot)

        ctc_setpoint = np.array([state[0], state[1], qd[0], qd[1]])
        tau = self._ctc.step(t, state, ctc_setpoint, robot)
        return tau


class CoordinatedController(Controller):
    def __init__(self, Kp, Kd):
        self._ctc = ComputedTorqueController(Kp, Kd)
        self._ijc = InverseJacobianController()

    def step(self, t, state, set_point, robot):
        """
        Find control signal for step at time t.
        
        Parameters
        ----------
        t : float
            time value
        state : numpy.ndarray[float]
            robot2R state = [q1, q2, q1d, q2d, qp, qpd]
        set_point : numpy.ndarray[float]
            desired velocity = [xd, yd]
        robot : Robot2R
            a 2R robot object

        Returns
        -------
        tau : float
            torque for robot joints
        """

        qd = self._ijc.step(t, state, set_point, robot)

        ctc_setpoint = np.array([state[0], state[1], qd[0], qd[1]])
        tau = self._ctc.step(t, state, ctc_setpoint, robot)
        return tau