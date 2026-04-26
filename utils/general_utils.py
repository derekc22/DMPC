import numpy as np

from utils.mj_utils import mj_cleanup
from utils.plot_utils import plot_t, plot_xyz, animate_xyz_gif


def cleanup(env_cfg,
            x_cl,
            u_cl,
            J_cl,
            wall_clk,
            name,
            fname,
            vis_cfg=None,
            qualifier="",
            normalize_J_by_M=False):
    J_avg = np.mean(J_cl)
    if normalize_J_by_M:
        J_avg /= env_cfg.M
    wall_clk = np.median(wall_clk)

    plot_t(env_cfg, x_cl, u_cl, J_avg, name, fname, qualifier)
    plot_xyz(env_cfg, x_cl, J_avg, wall_clk, name, fname, qualifier)
    animate_xyz_gif(env_cfg, x_cl, J_avg, wall_clk, name, fname, qualifier)

    if vis_cfg is not None:
        mj_cleanup(vis_cfg, name, fname, qualifier)
