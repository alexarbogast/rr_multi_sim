import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

HISTORY_LENGTH = 700

def plot_solution(sol):
    t = np.linspace(sol.t[0], sol.t[-1], 5000)
    cont = sol.sol(t)

    plt.style.use('dark_background')
    _, ax = plt.subplots()
    for c in cont:
        ax.plot(t, c.T)
    ax.set_xlabel('time (s)')
    ax.set_title('Robot Simulation Results')
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
    ax.set(xlim=(-3, 3), ylim=(-3, 3))
    
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

    ani = animation.FuncAnimation(fig, _animate, len(tt), interval=dt * 500, blit=True)
    plt.show()

def animateMulti(sol, robots):
    dt = 0.01
    tt = np.arange(sol.t[0], sol.t[-1], dt)
    state = sol.sol(tt)
    q1, q2 = state[:2].T, state[4:6].T

    fk1 = [robots[0].forward_kinematics(qq).T for qq in q1]
    fk2 = [robots[1].forward_kinematics(qq).T for qq in q2]

    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

    line1, = ax.plot([], [], 'o-', color='#81b1d2',lw=3) 
    line2, = ax.plot([], [], 'o-', color='#81b1d2',lw=3) 
    time_text = ax.text(0, 0, f'time: {tt[0]} s', transform=ax.transAxes)

    def _animate(i):
        p1 = fk1[i]
        line1.set_data(p1[0], p1[1])
        p2 = fk2[i]
        line2.set_data(p2[0], p2[1])
        time_text.set_text(f'time: {dt*i:0.2f} s')
        return line1, line2, time_text 
    
    ani = animation.FuncAnimation(fig, _animate, len(tt), interval=dt * 500, blit=True)
    plt.show()
