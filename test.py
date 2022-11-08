import numpy as np
from scipy.integrate import solve_ivp

from robot import Robot2R
from visualization import animate, plot_solution

def controller(t, x, robot):
    q = x[:2]
    
    setpoint = np.array([[-np.pi/2], [0]])
    error = setpoint - np.atleast_2d(q).T

    P_gains = np.diag([5, 5])

    control_input = (P_gains @ error)*0
    return control_input

def feedback_linearization(x, robot):
    q, qd = x[:2], x[2:]
    G, C = robot._G(q), robot._C(q, qd)
    M = robot._M(q)

    setpoint = np.array([0.1, 0.2])
    error = np.atleast_2d(setpoint - q)
    return 500*error + G.T + C.T

def robot2R_dynamics_sim(t_span, y0, robot):
    def _sim(t, state):
        # control
        control = controller(t, state, robot)

        # plant
        q, qd = state[:2], state[2:]
        qdd = robot.forward_dynamics(q, qd, control)
        return np.concatenate((qd, qdd.T[0]))

    return solve_ivp(_sim, t_span, y0, dense_output=True)

if __name__ == '__main__':
    robot = Robot2R(1, 1, 1, 1)

    t_span = [0, 10]
    y0 =  np.array([0, 0, 0, 0])
    
    # simulate robot
    #sol = solve_ivp(robot2R_sim, t_span, y0, dense_output=True, args=[robot])

    sol = robot2R_dynamics_sim(t_span, y0, robot)
    animate(sol, robot)
    #plot_solution(sol)