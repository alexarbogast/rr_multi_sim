import numpy as np
import matplotlib.pyplot as plt

from robot import Robot2R
from controller import Controller, ComputedTorqueController
from simulation import Simulation
from visualization import animate, plot_solution


if __name__ == '__main__':
    robot = Robot2R(1, 1, 1, 1)
    #control = Controller()
    control = ComputedTorqueController(5)

    # create trajectory
    q1, q2 = lambda t: 2*np.pi/10*t, lambda t: 2*np.pi/5*t
    q1d, q2d  =  lambda t: 2*np.pi/10, lambda t: 2*np.pi/5 
    q1dd, q2dd = lambda t: 0, lambda t: 0

    traj = lambda t: np.array([q1(t), q2(t), q1d(t), q2d(t), q1dd(t), q2dd(t)])

    sim = Simulation(robot, control)
    sol = sim.dynamics_sim(traj)
    animate(sol, robot)

    plot_solution(sol)