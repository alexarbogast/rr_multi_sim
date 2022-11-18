import numpy as np
import matplotlib.pyplot as plt

from robot import Robot2R, Positioner
from controller import Controller, TrajectoryController, CoordinatedRRController, PositionerController
from trajectory import Trajectory, Trajectory5PtP
from simulation import Simulation2R, SimulationMulti, SimulationMultiPositioner, SimulationMultiPositioner2
from visualization import animate2R, animateMulti, animateMultiPos, plot_solution

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
    Kd = 200*np.identity(2)
    control1 = TrajectoryController(Kp, Kd, Kp)
    control2 = TrajectoryController(Kp, Kd, Kp)
    pos_control = Controller()

    sim = SimulationMultiPositioner([robot1, robot2], [control1, control2],
                                     pos, pos_control)

    # trajectories
    points1 = np.array([[-0.2, -0.5], [-1.5, -0.5] , [-1.5, 0.5], [-0.2, 0.5], [-0.2, -0.5]])
    traj1 = Trajectory5PtP(points1, 0.5)

    points2 = np.array([[0.2, -0.5], [1.5, -0.5] , [1.5, 0.5], [0.2, 0.5], [0.2, -0.5]])
    traj2 = Trajectory5PtP(points2, 0.5)

    sol = sim.dynamics_sim([traj1, traj2])
    animateMultiPos(sol, [robot1, robot2], pos)

def positioner_multi_sim2():
    # mechanical units
    robot1 = Robot2R(1, 1, 1, 1, g=0, base=(-2, 0))
    robot2 = Robot2R(1, 1, 1, 1, g=0, base=( 2, 0))
    pos = Positioner(1, 1.5)

    # controllers
    Kp = 50*np.identity(2)
    Kd = 50*np.identity(2)
    #control1 = CoordinatedRRController(Kp, Kd, Kp)
    control1 = CoordinatedRRController(Kp, Kd, Kp)
    control2 = CoordinatedRRController(Kp, Kd, Kp)
    pos_control = PositionerController(1)

    sim = SimulationMultiPositioner2([robot1, robot2], [control1, control2],
                                     pos, pos_control)

    # trajectories
    path1 = np.array([[-0.5, -0.5], [-1.25, -0.5] , [-1.25, 0.5], [-0.5, 0.5], [-0.5, -0.5]])
    traj1 = Trajectory5PtP(path1, 0.5)

    path2 = np.array([[1.2, -0.3], [1.2, 0.3], [0.2, 0.3], [0.2, -0.3], [1.2, -0.3]])
    traj2 = Trajectory5PtP(path2, 0.5)

    sol = sim.dynamics_sim([traj1, traj2])
    animateMultiPos(sol, [robot1, robot2], pos, path1, path2)


if __name__ == '__main__':
    #robot2R_sim()
    #robot2R_multi_sim()
    #positioner_multi_sim()
    positioner_multi_sim2()