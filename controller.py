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


class CoordInverseJacobianController(Controller):
    def __init__(self, Kp):
        self._Kp = Kp

    def _get_params(self, q1, q2, qp, robot):
        """ Returns the robot and positioner Jacobians """

        l1, l2 = robot._l1, robot._l2
        xb, yb = robot._base
        sp, cp = np.sin(qp), np.cos(qp)
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)

        # find robot jacobian
        jr11 =  sp*(c1*l1 + c12*l2) - cp*(s1*l1 + s12*l2)
        jr12 =  l2*(c12*sp - s12*cp)
        jr21 =  cp*(c1*l1 + c12*l2) + sp*(s1*l1 + l2*s12)
        jr22 =  l2*(c12*cp + s12*sp)
        Jr=  np.array([[jr11, jr12], [jr21, jr22]])

        # find positioner jacobian
        jp1 = cp*yb - sp*xb - sp*(c1*l1 + c12*l2) + cp*(l1*s1 + l2*s12)
        jp2 = -cp*xb - sp*yb - sp*(s1*l1 + s12*l2) - cp*(l1*c1 + l2*c12)
        Jp = np.array([jp1, jp2])
        return Jr, Jp

    def _Tpw(self, qp):
        """ Homogeneous transformation from world to positioner frames"""

        sp, cp = np.sin(qp), np.cos(qp)
        Tpw = np.array([[ cp, sp, 0],
                        [-sp, cp, 0],
                        [ 0,  0,  1]])
        return Tpw

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
            desired position and velocity = [x_d, y_d, xd_d, yd_d, qpd]
            in positioner frame
        robot : Robot2R
            a 2R robot object

        Returns
        -------
        qd : numpy.array[float]
            joint velocity
        """

        qr, qp = state[:2], state[4]
        vd = set_point[2:4]
        qpd = set_point[4]

        # robot and positioner Jacobians
        Jr, Jp = self._get_params(qr[0], qr[1], qp, robot)

        *_, pd = robot.forward_kinematics(qr)
        ep = set_point[:2] - (self._Tpw(qp) @ np.append(pd, 1))[:-1]
        uk = self._Kp @ ep + vd
        u = np.linalg.inv(Jr) @ (uk - Jp*qpd)
        return u


class CoordinatedRRController(Controller):
    def __init__(self, Kp, Kd, Kp_jac):
        self._ctc = ComputedTorqueController(Kp, Kd)
        self._cijc = CoordInverseJacobianController(Kp_jac)

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
            desired position and velocity = [x_d, y_d, xd_d, yd_d, qpd]
            in positioner frame
        robot : Robot2R
            a 2R robot object

        Returns
        -------
        tau : numpy.array[float]
            torque for robot joints
        """

        qd = self._cijc.step(t, state, set_point, robot)
        ctc_setpoint = np.array([state[0], state[1], qd[0], qd[1]])
        tau = self._ctc.step(t, state[:4], ctc_setpoint, robot)
        return tau


class PositionerController(Controller):
    def __init__(self, Kd):
        self._Kd = Kd

    def step(self, t, state, set_point, robot):
        """
        Find control signal for step at time t.
        
        Parameters
        ----------
        t : float
            time value
        state : numpy.ndarray[float]
            positioner state = [qp, qpd]
        set_point : float
            desired velocity = qpd_d
        robot : Positioner
            a positioner object

        Returns
        -------
        tau : float
            torque for robot joints
        """

        # velocity control
        ed = set_point - state[-1]
        u = self._Kd * ed
        tau = robot._I*(u)

        # position control
        #e = set_point - state[0]
        #u = self._Kd * e
        #tau = robot._I*(u)
        return tau


class MaxManipulabilityController(Controller):
    def __init__(self, Kd):
        self._pc = PositionerController(Kd)

    def _gradient(self, state, robots):
        r1, r2 = robots[0], robots[1]
        q2_1, q2_2 = state[1], state[3]
        
        del_psi1 = -2*(r1._l1**2)*(r1._l2**2)*np.sin(q2_1)*np.cos(q2_1)
        del_psi2 = -2*(r2._l1**2)*(r2._l2**2)*np.sin(q2_2)*np.cos(q2_2)

        return del_psi1 + del_psi2

    def _gradient_approx(self, state, robots):
        r1, r2 = robots[0], robots[1]


    def step(self, t, state, robots, positioner):
        """
        Find control signal for step at time t.

        This controller adjusts the positioner velocity to maxmimize the 
        combined manipulability of each robot
        
        Parameters
        ----------
        t : float
            time value
        state : numpy.ndarray[float]
            state = [q11, q21, qd11, qd21, q12, q22, qd12, qd22, qp, qpd]
        set_point : float
            desired velocity = qpd_d
        robots : Positioner
            a positioner object

        Returns
        -------
        tau : float
            torque for robot joints
        """
        
        eta = 1.5
        gradient = self._gradient(state, robots)

        qp_old = state[8]
        qp_new = qp_old + eta*gradient
        qp_d = eta*gradient
        #print(qp_d)

        # position control
        pos_state = state[8:]
        tau = self._pc.step(t, pos_state, qp_d, positioner)
        return tau, qp_d


class CenterRobotsController(Controller):
    def __init__(self, Kd):
        self._pc = PositionerController(Kd)

    def step(self, t, state, robots, positioner):
        """
        Find control signal for step at time t.
        
        This adjusts the positioner velocity to keep the robots close to their
        home positions

        Parameters
        ----------
        t : float
            time value
        state : numpy.ndarray[float]
            state = [q11, q21, qd11, qd21, q12, q22, qd12, qd22, qp, qpd]
        set_point : float
            desired velocity = qpd_d
        robots : Positioner
            a positioner object

        Returns
        -------
        tau : float
            torque for robot joints
        """

        v_base1 = np.array(robots[0]._base)
        v_base2 = np.array(robots[1]._base)

        *_, p1 = robots[0].forward_kinematics(state[0:2])
        *_, p2 = robots[1].forward_kinematics(state[4:6])

        #print(v_base1)
        #print(p1)
        #print(v_base2)
        #print(p2)
        #print()

        v_base1 = v_base1 / np.linalg.norm(v_base1)
        v_base2 = v_base2 / np.linalg.norm(v_base2)
        p1 = p1 / np.linalg.norm(p1)
        p2 = p2 / np.linalg.norm(p2)

        e1 = np.cross(p1, v_base1)
        e2 = np.cross(p2, v_base2)

        K = 1
        e = e1 + e2
        qp_d = K * e

        # position control
        pos_state = state[8:]
        tau = self._pc.step(t, pos_state, qp_d, positioner)
        return tau, qp_d
