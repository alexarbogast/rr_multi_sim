import numpy as np
import matplotlib.pyplot as plt

from robot import Robot2R, Positioner
from controller import *
from trajectory import Trajectory, Trajectory5PtP
from simulation import Simulation2R, SimulationMulti, SimulationMultiPositioner
from visualization import *


def robot2R_sim():
    robot = Robot2R(1, 1, 1, 1)

    Kp = 50*np.identity(2)
    Kd = 50*np.identity(2)
    control = TrajectoryController(Kp, Kd, Kp)
    
    points = np.array([[0.4, 0.4], [1.2, 0.4] , [1.2, 1.2], [0.4, 1.2], [0.4, 0.4]])
    traj = Trajectory(points, 0.5, 0)
    traj = Trajectory5PtP(points, 0.50)

    sim = Simulation2R(robot, control)
    sol = sim.dynamics_sim(traj)
    animate2R(sol, robot)
    plot_solution(sol)

def robot2R_multi_sim():
    robot1 = Robot2R(1, 1, 1, 1, base=(-1, 0))
    robot2 = Robot2R(1, 1, 1, 1, base=(1, 0))

    # controllers
    Kp = 50*np.identity(2)
    Kd = 50*np.identity(2)

    control1 = TrajectoryController(Kp, Kd, Kp)
    control2 = TrajectoryController(Kp, Kd, Kp)
    sim = SimulationMulti([robot1, robot2], [control1, control2])

    # trajectories
    path1 = np.array([[-0.5, -0.5], [-1.25, -0.5] , [-1.25, 0.5], [-0.5, 0.5], [-0.5, -0.5]])
    traj1 = Trajectory5PtP(path1, 0.5)

    path2 = np.array([[1.2, -0.3], [1.2, 0.3], [0.2, 0.3], [0.2, -0.3], [1.2, -0.3]])
    traj2 = Trajectory5PtP(path2, 0.5)

    sol = sim.dynamics_sim([traj1, traj2])
    animateMulti(sol, [robot1, robot2])

def positioner_multi_sim():
    # mechanical units
    robot1 = Robot2R(1, 1, 1, 1, g=0, base=(-2, 0))
    robot2 = Robot2R(1, 1, 1, 1, g=0, base=( 2, 0))
    pos = Positioner(1, 1.5)

    # controllers
    Kp = 100*np.identity(2)
    Kd = 100*np.identity(2)

    control1 = CoordinatedRRController(Kp, Kd, Kp)
    control2 = CoordinatedRRController(Kp, Kd, Kp)
    pos_control = CenterRobotsController(5)

    sim = SimulationMultiPositioner([robot1, robot2], [control1, control2],
                                     pos, pos_control)

    # trajectories
    path1 = np.array([[-0.5, -0.5], [-1.25, -0.5] , [-1.25, 0.5], [-0.5, 0.5], [-0.5, -0.5]])
    path2 = np.array([[1.2, -0.3], [1.2, 0.3], [0.2, 0.3], [0.2, -0.3], [1.2, -0.3]])
    
    #path1 = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    #path2 = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    
    traj1 = Trajectory5PtP(path1, 0.5)
    traj2 = Trajectory5PtP(path2, 0.5)

    sol = sim.dynamics_sim([traj1, traj2])
    animateMultiPos(sol, [robot1, robot2], pos, path1, path2)
#
    #legend = (['q11', 'q21', 'qd11', 'qd21', 'q12', 
    #           'q22', 'qd12', 'qd22', 'qp', 'qpd'])
    #plot_solution(sol, legend)
    #plot_manipulability(sol, [robot1, robot2])
    plot_coordinated_error(sol, [robot1, robot2], [traj1, traj2])

if __name__ == '__main__':
    #robot2R_sim()
    #robot2R_multi_sim()
    positioner_multi_sim()

    #p = np.array([np.array([0, 0]), np.array([1, 1]), np.array([2, 2])])
    #act = np.array([np.array([0.1, 0.1]), np.array([1.2, 1.2]), np.array([2.3, 2.3])])

    #e = p - act
    #print(np.linalg.norm(e, axis=1))