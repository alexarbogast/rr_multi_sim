import numpy as np
import matplotlib.pyplot as plt

from robot import Robot2R, Positioner
from controller import Controller, TrajectoryController
from simulation import Simulation2R, SimulationMulti, SimulationMultiPositioner
from visualization import animate2R, animateMulti, animateMultiPos, plot_solution

def robot2R_sim():
    robot = Robot2R(1, 1, 1, 1)

    Kp = 1*np.identity(2)
    Kd = 100*np.identity(2)
    control = TrajectoryController(Kp, Kd, Kp)
    
    # create trajectory
    #q1, q2 = lambda t: 2*np.pi/10*t, lambda t: 2*np.pi/5*t
    #q1d, q2d  =  lambda t: 2*np.pi/10, lambda t: 2*np.pi/5 
    #q1dd, q2dd = lambda t: 0, lambda t: 0
#
    #traj = lambda t: np.array([q1(t), q2(t), q1d(t), q2d(t), q1dd(t), q2dd(t)])

    # square trajectory
    time = 8;
    time_per_side = time / 4;
    def traj(t):
        speed = 1 / (time_per_side)
        if t < time_per_side*1:
            return np.array([0.4+t*speed, 0.4, speed, 0.0])
        elif t < time_per_side*2:
            return np.array([1.4, 0.4+(t-time_per_side)*speed, 0.0, speed])
        elif t < time_per_side*3:
            return np.array([1.4-(t-2*time_per_side)*speed, 1.4, -speed, 0.0])
        else:
            return np.array([0.4, 1.4-(t-3*time_per_side)*speed, 0, -speed])
    
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

def positioner_multi_sim():
    # mechanical units
    robot1 = Robot2R(1, 1, 1, 1, g=0, base=(-2, 0))
    robot2 = Robot2R(1, 1, 1, 1, g=0, base=( 2, 0))
    pos = Positioner(1, 1.5)

    # controllers
    Kp = 5*np.identity(2)
    Kd = 50*np.identity(2)
    control1 = TrajectoryController(Kp, Kd)
    control2 = TrajectoryController(Kp, Kd)
    pos_control = Controller()

    sim = SimulationMultiPositioner([robot1, robot2], [control1, control2],
                                     pos, pos_control)

    # trajectories
    time = 10 / 4;
    def traj1(t):
        speed = 0.3
        if t < time*1:
            return np.array([speed, 0.0])
        elif t < time*2:
            return np.array([0, speed])
        elif t < time*3:
            return np.array([-speed, 0.0])
        else:
            return np.array([0, -speed])

    def traj2(t):
        speed = 0.3
        if t < time*1:
            return np.array([-speed, 0.0])
        elif t < time*2:
            return np.array([0, speed])
        elif t < time*3:
            return np.array([speed, 0.0])
        else:
            return np.array([0, -speed])

    sol = sim.dynamics_sim([traj1, traj2])
    animateMultiPos(sol, [robot1, robot2], pos)

if __name__ == '__main__':
    robot2R_sim()
    #robot2R_multi_sim()
    #positioner_multi_sim()