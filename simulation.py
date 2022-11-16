import numpy as np
from scipy.integrate import solve_ivp

class Simulation2R:
    def __init__(self, robot, controller):
        self._robot = robot
        self._controller = controller

    def _sim_step(self, t, state, trajectory):
        # control
        set_point = trajectory(t)
        control_input = self._controller.step(t, state, set_point, self._robot)

        # plant
        q, qd = state[:2], state[2:]
        qdd = self._robot.forward_dynamics(q, qd, control_input)
        return np.concatenate((qd, qdd.T[0]))

    def dynamics_sim(self, trajectory=None):
        t_span = [0, 8]

        p0 =  np.array([0, 0])
        if trajectory is not None:
            p0 = self._robot.inverse_kinematics(trajectory(0)[:2])
        y0 = np.concatenate((p0, [0, 0]))
        
        return solve_ivp(self._sim_step, t_span, y0, 
                         dense_output=True, args=[trajectory])


class SimulationMulti:
    def __init__(self, robots, controllers):
        self._robots = robots
        self._controllers = controllers

    def _sim_step(self, t, state):
        # control

        # plant
        q1, qd1 = state[:2], state[2:4]
        q2, qd2 = state[4:6], state[6:]
        qdd1 = self._robots[0].forward_dynamics(q1, qd1, 0)
        qdd2 = self._robots[1].forward_dynamics(q2, qd2, 0)
        return np.concatenate((qd1, qdd1.T[0], qd2, qdd2.T[0]))

    def dynamics_sim(self):
        t_span = [0, 10]
        y0 =  np.zeros(8)
        
        return solve_ivp(self._sim_step, t_span, y0, dense_output=True)


class SimulationMultiPositioner:
    def __init__(self, robots, rob_controllers, positioner, pos_controller):
        self._robots = robots
        self._controller = rob_controllers
        self._positioner = positioner
        self._pos_controller = pos_controller

    def _sim_step(self, t, state, trajectories):
        rob1_state, rob2_state = state[:4], state[4:8]
        pos_state = state[8:]

        # control
        set_point1 = trajectories[0](t)
        set_point2 = trajectories[1](t)
        control_input1 = self._controller[0].step(t, rob1_state, set_point1, self._robots[0])
        control_input2 = self._controller[1].step(t, rob2_state, set_point2, self._robots[1])

        # plant
        q1, qd1 = rob1_state[:2], rob1_state[2:]
        q2, qd2 = rob2_state[:2], rob2_state[2:]
        qp, qpd = pos_state[0], pos_state[1]

        qdd1 = self._robots[0].forward_dynamics(q1, qd1, control_input1)
        qdd2 = self._robots[1].forward_dynamics(q2, qd2, control_input2)
        qpdd = self._positioner.forward_dynamics(qp, qpd, 0)
        return np.concatenate((qd1, qdd1.T[0], 
                               qd2, qdd2.T[0], 
                               np.array([qpd, qpdd])))

    def dynamics_sim(self, trajectories):
        t_span = [0, 10]
        
        q1i = np.array([np.pi/3, -2*np.pi/3, 0, 0])
        q2i = np.array([2*np.pi/3, 2*np.pi/3, 0, 0])
        qpi = np.array([0, np.pi/2])
        y0 = np.concatenate((q1i, q2i, qpi))

        return solve_ivp(self._sim_step, t_span, y0, dense_output=True,
                         args=[trajectories])