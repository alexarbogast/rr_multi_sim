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

        #y0 =  np.array([0, 0, 0, 0])
        #if trajectory is not None:
        #    y0 = trajectory(0)[:4]

        y0 = np.array([np.pi/2, -np.pi*3/4, 0, 0])
        
        return solve_ivp(self._sim_step, t_span, y0, 
                         dense_output=True, args=[trajectory])


class SimulationMulti:
    def __init__(self, robots, controllers):
        self._robots = robots
        self._controllers = controllers

    def _sim_step(self, t, state):
        # control

        #plant
        q1, qd1 = state[:2], state[2:4]
        q2, qd2 = state[4:6], state[6:]
        qdd1 = self._robots[0].forward_dynamics(q1, qd1, 0)
        qdd2 = self._robots[1].forward_dynamics(q2, qd2, 0)
        return np.concatenate((qd1, qdd1.T[0], qd2, qdd2.T[0]))

    def dynamics_sim(self):
        t_span = [0, 10]
        y0 =  np.zeros(8)
        
        return solve_ivp(self._sim_step, t_span, y0, dense_output=True)