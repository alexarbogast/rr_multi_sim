import numpy as np
import matplotlib.pyplot as plt

from robot import Robot2R
from controller import Controller, ComputedTorqueController, CombinedController
from simulation import Simulation2R, SimulationMulti
from visualization import animate2R, animateMulti, plot_solution

def robot2R_sim():
    robot = Robot2R(1, 1, 1, 1)

    Kp = 5*np.identity(2)
    Kd = 50*np.identity(2)
    control = CombinedController(Kp, Kd)
    
    # create trajectory
    #q1, q2 = lambda t: 2*np.pi/10*t, lambda t: 2*np.pi/5*t
    #q1d, q2d  =  lambda t: 2*np.pi/10, lambda t: 2*np.pi/5 
    #q1dd, q2dd = lambda t: 0, lambda t: 0
#
    #traj = lambda t: np.array([q1(t), q2(t), q1d(t), q2d(t), q1dd(t), q2dd(t)])

    # square trajectory
    time = 8 / 4;
    def traj(t):
        speed = 0.4
        if t < time*1:
            return np.array([speed, 0.0])
        elif t < time*2:
            return np.array([0, speed])
        elif t < time*3:
            return np.array([-speed, 0.0])
        else:
            return np.array([0, -speed])
    
    sim = Simulation2R(robot, control)
    sol = sim.dynamics_sim(traj)
    animate2R(sol, robot)

def robot2R_multi_sim():
    robot1 = Robot2R(1, 1, 1, 1, base=(-1, 0))
    robot2 = Robot2R(1, 1, 1, 1, base=(1, 0))

    control = Controller()

    sim = SimulationMulti([robot1, robot2], [control, control])
    sol = sim.dynamics_sim()
    animateMulti(sol, [robot1, robot2])

if __name__ == '__main__':
    robot2R_sim()
    #robot2R_multi_sim()
    
    #plt.style.use('dark_background')
    #print(plt.rcParams['axes.prop_cycle'])