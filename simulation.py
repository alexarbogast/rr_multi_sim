import numpy as np
from scipy.integrate import solve_ivp

class Simulation:
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
        t_span = [0, 10]

        y0 =  np.array([0, 0, 0, 0])
        if trajectory is not None:
            y0 = trajectory(0)[:4]
        
        return solve_ivp(self._sim_step, t_span, y0, 
                         dense_output=True, args=[trajectory])