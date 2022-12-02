import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

HISTORY_LENGTH = 700

def plot_solution(sol, labels=None):
    t = np.linspace(sol.t[0], sol.t[-1], 5000)
    cont = sol.sol(t)

    plt.style.use('dark_background')
    _, ax = plt.subplots()

    for c in cont:
        ax.plot(t, c.T)
    if labels is not None:
        ax.legend(labels)

    ax.set_xlabel('time (s)')
    ax.set_title('Robot Simulation Results')
    plt.show()

def plot_manipulability(sol, robots):
    t = np.linspace(sol.t[0], sol.t[-1], 1000)
    cont = sol.sol(t)

    q1 = cont.T[:, 0:2]
    q2 = cont.T[:, 4:6]

    J1 = [robots[0].jacobian(q) for q in q1]
    J2 = [robots[1].jacobian(q) for q in q2]

    u1 = [np.sqrt(np.linalg.det(J @ J.T)) for J in J1]
    u2 = [np.sqrt(np.linalg.det(J @ J.T)) for J in J2]
    u_sum = [x + y for x, y in zip(u1, u2)]

    plt.style.use('dark_background')
    plt.plot(t, u1)
    plt.plot(t, u2)
    plt.plot(t, u_sum)
    plt.show()

def plot_coordinated_error(sol, robots, trajectories):
    def rotate_point(p, pos_angle):
        R = np.array([[np.cos(pos_angle),  np.sin(pos_angle)],
                      [-np.sin(pos_angle),  np.cos(pos_angle)]])
        return R @ p
    
    n = len(robots)
    assert n == len(trajectories), \
        'Number of robots and trajectories do not match'

    dt = 0.01
    tt = np.arange(sol.t[0], sol.t[-1], dt)
    state = sol.sol(tt)
    q1, q2, qp = state[:2].T, state[4:6].T, state[8]
    qs = [q1, q2]

    plt.style.use('dark_background')
    for i in range(n):
        p_des = np.array([trajectories[i].get_point(t)[0] for t in tt])
        #p_act = np.array([robots[i].forward_kinematics(q)[2] for q in qs[i]])
        p_act = []
        for q, pos_angle in zip(qs[i], qp):
            p = robots[i].forward_kinematics(q)[2]
            p_act.append(rotate_point(p, pos_angle))
        p_act = np.array(p_act)

        e = np.linalg.norm(p_des - p_act, axis=1)
        plt.plot(tt, e)

    plt.show()

def animate2R(sol, robot):
    dt = 0.01
    tt = np.arange(sol.t[0], sol.t[-1], dt)
    state = sol.sol(tt)
    q = state[:2].T

    fk = [robot.forward_kinematics(qq).T for qq in q]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set(xlim=(-2, 2), ylim=(-2, 2))
    
    line, = ax.plot([], [], 'o-', color='#81b1d2',lw=4) 
    trace, = ax.plot([], [], '.-', color='orange', lw=0.5, ms=2)
    time_text = ax.text(0, 0, f'time: {tt[0]} s', transform=ax.transAxes)
    history_x, history_y = deque(maxlen=HISTORY_LENGTH), deque(maxlen=HISTORY_LENGTH)

    def _animate(i):
        p = fk[i]
        line.set_data(p[0], p[1])
        time_text.set_text(f'time: {dt*i:0.2f} s')

        history_x.appendleft(p[0][-1])
        history_y.appendleft(p[1][-1])
        trace.set_data(history_x, history_y)
        return line, trace, time_text

    anim = animation.FuncAnimation(fig, _animate, len(tt), interval=dt * 500, blit=True)
    
    #writervideo = animation.FFMpegWriter(fps=60)
    #anim.save('animation2.mp4', writer=writervideo)
    plt.show()

def animateMulti(sol, robots):
    dt = 0.01
    tt = np.arange(sol.t[0], sol.t[-1], dt)
    state = sol.sol(tt)
    q1, q2 = state[:2].T, state[4:6].T

    fk1 = [robots[0].forward_kinematics(qq).T for qq in q1]
    fk2 = [robots[1].forward_kinematics(qq).T for qq in q2]

    history1_x, history1_y = deque(maxlen=HISTORY_LENGTH), deque(maxlen=HISTORY_LENGTH)
    history2_x, history2_y = deque(maxlen=HISTORY_LENGTH), deque(maxlen=HISTORY_LENGTH)

    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

    line1, = ax.plot([], [], 'o-', color='#81b1d2',lw=3) 
    line2, = ax.plot([], [], 'o-', color='#81b1d2',lw=3)
    trace1, = ax.plot([], [], '.-', color='orange', lw=0.5, ms=2)
    trace2, = ax.plot([], [], '.-', color='orange', lw=0.5, ms=2)
    time_text = ax.text(0, 0, f'time: {tt[0]} s', transform=ax.transAxes)

    def _animate(i):
        p1 = fk1[i]
        line1.set_data(p1[0], p1[1])
        p2 = fk2[i]
        line2.set_data(p2[0], p2[1])
        time_text.set_text(f'time: {dt*i:0.2f} s')

        history1_x.appendleft(p1[0][-1])
        history1_y.appendleft(p1[1][-1])
        history2_x.appendleft(p2[0][-1])
        history2_y.appendleft(p2[1][-1])
        trace1.set_data(history1_x, history1_y)
        trace2.set_data(history2_x, history2_y)
        return line1, line2, trace1, trace2, time_text 
    
    ani = animation.FuncAnimation(fig, _animate, len(tt), interval=dt * 500, blit=True)
    plt.show()

def animateMultiPos(sol, robots, positioner, path1, path2):
    dt = 0.01
    tt = np.arange(sol.t[0], sol.t[-1], dt)
    state = sol.sol(tt)
    q1, q2, qp = state[:2].T, state[4:6].T, state[8]

    # find forward kinematics for all robot configurations
    fk1 = [robots[0].forward_kinematics(qq).T for qq in q1]
    fk2 = [robots[1].forward_kinematics(qq).T for qq in q2]

    # positioner axes for all positioner angles
    scale = positioner._r * 0.3
    coord_axes = np.identity(2)
    Rot = lambda q: np.array([[np.cos(q), -np.sin(q)], [np.sin(q), np.cos(q)]])
    Fpos = [scale * Rot(q) @ coord_axes for q in qp]

    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set(xlim=(-3, 3), ylim=(-2, 2))

    # initialize robot lines
    line1, = ax.plot([], [], 'o-', color='#81b1d2',lw=3) # robot 1
    line2, = ax.plot([], [], 'o-', color='#81b1d2',lw=3) # robot 2
    time_text = ax.text(0, 0, f'time: {tt[0]} s', transform=ax.transAxes)

    # initialize positioner
    circle = plt.Circle((0, 0), positioner._r, fill=False)
    ax.add_patch(circle)
    axis_x, = ax.plot([], [], color='r',lw=2) 
    axis_y, = ax.plot([], [], color='b',lw=2)

    # TEMPORARY
    paths1, = ax.plot([], [], color='y',lw=2) 
    rot_paths1 = [np.einsum('ij, kj->ki', Rot(q), path1) for q in qp]

    paths2, = ax.plot([], [], color='g',lw=2) 
    rot_paths2 = [np.einsum('ij, kj->ki', Rot(q), path2) for q in qp]

    def _animate(i):
        p1 = fk1[i]
        line1.set_data(p1[0], p1[1])
        p2 = fk2[i]
        line2.set_data(p2[0], p2[1])
        time_text.set_text(f'time: {dt*i:0.2f} s')

        coords = Fpos[i]
        axis_x.set_data([0, coords[0, 0]], [0, coords[1, 0]])
        axis_y.set_data([0, coords[0, 1]], [0, coords[1, 1]])

        # TEMPORARY
        rot_path1 = rot_paths1[i]
        paths1.set_data(rot_path1.T[0], rot_path1.T[1]) 
        rot_path2 = rot_paths2[i]
        paths2.set_data(rot_path2.T[0], rot_path2.T[1]) 
        return line1, line2, time_text, axis_x, axis_y, paths1, paths2
    
    anim = animation.FuncAnimation(fig, _animate, len(tt), interval=dt * 500, blit=True)
    
    writervideo = animation.FFMpegWriter(fps=120)
    #anim.save('animation3.mp4', writer=writervideo)
    plt.show()
