import matplotlib.pyplot as plt
import numpy as np
import math
import os
import imageio

def plot_t(t_max, N, M, x_cl, u_cl, J_cl_avg, fname, qualifier=""):

    os.makedirs("plots", exist_ok=True)

    # internal layout parameters
    max_width = 3          # maximum number of subplots per row
    col_scale = 6          # width scaling per column
    row_scale = 3.5        # height scaling per row

    t_x = np.linspace(0, t_max, N + 1)
    t_u = np.linspace(0, t_max, N)

    nx = x_cl.shape[1]
    nu = u_cl.shape[1]

    # ------------------------------------------------------------
    # 1. STATES
    # ------------------------------------------------------------
    cols = min(max_width, nx)
    rows = math.ceil(nx / max_width)

    fig, axs = plt.subplots(rows, cols, figsize=(col_scale * cols, row_scale * rows))
    axs = np.array(axs).reshape(rows, cols)

    for i in range(nx):
        r = i // cols
        c = i % cols
        ax = axs[r, c]

        for m in range(M):
            ax.plot(t_x, x_cl[m, i, :], label=f'agent {m}')

        ax.set_xlabel('t [s]')
        ax.set_ylabel(f'x[{i}]')
        ax.grid()
        if i == 0:
            ax.legend()

    plt.suptitle(f'{fname} mpc x_cl, Jbar = {J_cl_avg:.3f}')
    plt.tight_layout()
    plt.savefig(f"figures{fname}_mpc{('_' + qualifier) if qualifier else ''}_x_cl.png")
    plt.close()

    # ------------------------------------------------------------
    # 2. CONTROLS
    # ------------------------------------------------------------
    cols = min(max_width, nu)
    rows = math.ceil(nu / max_width)

    fig, axs = plt.subplots(rows, cols, figsize=(col_scale * cols, row_scale * rows))
    axs = np.array(axs).reshape(rows, cols)

    for i in range(nu):
        r = i // cols
        c = i % cols
        ax = axs[r, c]

        for m in range(M):
            ax.step(t_u, u_cl[m, i, :], where='post', label=f'agent {m}')

        ax.set_xlabel('t [s]')
        ax.set_ylabel(f'u[{i}]')
        ax.grid()
        if i == 0:
            ax.legend()

    plt.suptitle(f'{fname} mpc u_cl, Jbar = {J_cl_avg:.3f}')
    plt.tight_layout()
    plt.savefig(f"figures{fname}_mpc{('_' + qualifier) if qualifier else ''}_u_cl.png")
    plt.close()






def plot_xyz(M, x_cl, x0_val, xf_val, J_cl_avg, obs, fname, qualifier=""):

    os.makedirs("plots", exist_ok=True)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    # trajectories
    for m in range(M):
        x = x_cl[m, 0, :]
        y = x_cl[m, 1, :]
        z = x_cl[m, 2, :]

        line, = ax.plot3D(x, y, z, label=f"agent {m}", marker="o", linestyle="-")#, markersize=3)
        color = line.get_color()
        ax.scatter(x0_val[m, 0], x0_val[m, 1], x0_val[m, 2], c=color, marker=".", s=200, edgecolors="black")
        if not "client" in fname:
            ax.scatter(xf_val[m, 0], xf_val[m, 1], xf_val[m, 2], c=color, marker="*", s=300, edgecolors="black")
        else:
            if m == 0:
                ax.scatter(xf_val[0], xf_val[1], xf_val[2], c=color, marker="*", s=300, edgecolors="black")

    # draw all obstacles
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)

    for o in obs:
        xo, yo, zo, ro = o

        x_s = xo + ro * np.outer(np.cos(u), np.sin(v))
        y_s = yo + ro * np.outer(np.sin(u), np.sin(v))
        z_s = zo + ro * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(x_s, y_s, z_s, alpha=0.4, color='gray')

    # enforce equal axis scaling in 3D
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    ax.grid()

    plt.title(f'{fname} mpc, Jbar = {J_cl_avg:.3f}')
    plt.tight_layout()
    plt.savefig(f"figures{fname}_mpc{('_' + qualifier) if qualifier else ''}_xyz.png")
    plt.close()






def animate_xyz_gif(M, x_cl, x0_val, xf_val, J_cl_avg, obs, fname, qualifier="", fps=10):

    os.makedirs("plots", exist_ok=True)

    frames = []
    T = x_cl.shape[2]

    # Precompute global axis limits for consistent scaling
    x_all = x_cl[:, 0, :].ravel()
    y_all = x_cl[:, 1, :].ravel()
    z_all = x_cl[:, 2, :].ravel()

    x_limits = (x_all.min(), x_all.max())
    y_limits = (y_all.min(), y_all.max())
    z_limits = (z_all.min(), z_all.max())

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    for k in range(T):

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        # trajectories
        for m in range(M):
            x = x_cl[m, 0, :k+1]
            y = x_cl[m, 1, :k+1]
            z = x_cl[m, 2, :k+1]

            line, = ax.plot3D(x, y, z, label=f"agent {m}", marker="o", linestyle="-")
            color = line.get_color()
            ax.scatter(x0_val[m, 0], x0_val[m, 1], x0_val[m, 2], c=color, marker=".", s=200, edgecolors="black")
            if not "client" in fname:
                ax.scatter(xf_val[m, 0], xf_val[m, 1], xf_val[m, 2], c=color, marker="*", s=300, edgecolors="black")
            else:
                if m == 0:
                    ax.scatter(xf_val[0], xf_val[1], xf_val[2], c=color, marker="*", s=300, edgecolors="black")
                
        # draw all obstacles
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)

        for o in obs:
            xo, yo, zo, ro = o
            x_s = xo + ro * np.outer(np.cos(u), np.sin(v))
            y_s = yo + ro * np.outer(np.sin(u), np.sin(v))
            z_s = zo + ro * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(x_s, y_s, z_s, alpha=0.4, color="gray")

        # enforce equal axis scaling
        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend()
        ax.grid()

        plt.title(f"{fname} mpc, Jbar = {J_cl_avg:.3f}")
        plt.tight_layout()

        frame_path = f"figures_frame_{k:05d}.png"
        plt.savefig(frame_path)
        frames.append(imageio.imread(frame_path))
        plt.close(fig)
        os.remove(frame_path)

    gif_path = f"figures{fname}_mpc{('_' + qualifier) if qualifier else ''}_xyz.gif"
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    
