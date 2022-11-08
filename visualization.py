import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

def animate(sol, robot):
    dt = 0.01
    tt = np.arange(sol.t[0], sol.t[-1], dt)
    state = sol.sol(tt)
    q = state[:2].T
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set(xlim=(-3, 3), ylim=(-3, 3))
    
    line, = ax.plot([], [], 'o-', lw=3) 
    time_text = ax.text(0, 0, f'time: {tt[0]} s', transform=ax.transAxes)

    fk = [robot.forward_kinematics(qq).T for qq in q]

    def _animate(i):
        p = fk[i]
        line.set_data(p[0], p[1])
        time_text.set_text(f'time: {dt*i:0.2f} s')
        return line, time_text

    ani = animation.FuncAnimation(fig, _animate, len(tt), interval=dt * 500, blit=True)
    plt.show()