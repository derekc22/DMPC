import numpy as np
import casadi as ca
from utils.plot_utils import *
from utils.mj_utils import mj_vis_step, mj_cleanup

def decentralized(dyn_cfg, dmpc_cfg, env_cfg, use_mj=False, vis_cfg=None):
    
    # parse configs
    name, f_plant, f_true, nx, nu, u_lim = (
        dyn_cfg.name,
        dyn_cfg.f_plant,
        dyn_cfg.f_true,
        dyn_cfg.nx,
        dyn_cfg.nu,
        dyn_cfg.u_lim,
    )
    if use_mj:
        mj_model, mj_data = dyn_cfg.mj_model, dyn_cfg.mj_data

    N, Q, R, H, term, mode = (
        dmpc_cfg.N,
        dmpc_cfg.Q,
        dmpc_cfg.R,
        dmpc_cfg.H,
        dmpc_cfg.term,
        dmpc_cfg.mode
    )

    T, dt, M, d_min, x0_val, obs, xf_val = (
        env_cfg.T,
        env_cfg.dt,
        env_cfg.M,
        env_cfg.d_min,
        env_cfg.x0_val,
        env_cfg.obs,
        env_cfg.xf_val,
    )
    
    # helpers
    def shift_pred(X):
        return np.hstack([X[:, 1:], X[:, -1:]])

    def set_xyz_others(m):
        i = 0
        for j in range(M):
            if j == m:
                continue
            agents[m]["opti"].set_value(agents[m]["XYZ_others"][i], pred_X[j][0:3, :])
            i += 1
    
    assert mode in ("gauss-seidel", "jacobi"), f"Invalid mode: {mode}"

    pred_X = np.zeros((M, nx, N + 1))
    pred_U = np.zeros((M, nu, N))

    # build a local OCP for one agent, with other agents' XYZ as parameters
    def build_agent_opti(m):
        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        x0 = opti.parameter(nx, 1)
        xf = opti.parameter(nx, 1)

        # set final state
        opti.set_value(xf, xf_val[m, :].reshape((nx, 1)))
        
        # control bounds and initial condition constraint
        opti.subject_to(X[:, 0] == x0)
        for i in range(nu):
            opti.subject_to(opti.bounded(u_lim[i][0], U[i, :], u_lim[i][1]))

        # obstacle constraint, center (xo,yo,zo), radius ro
        for o in obs:
            xo, yo, zo, ro = o
            opti.subject_to((X[0, :] - xo) ** 2 + (X[1, :] - yo) ** 2 + (X[2, :] - zo) ** 2 >= ro ** 2)

        # other agents' predicted positions over horizon
        XYZ_others = [opti.parameter(3, N + 1) for _ in range(M - 1)]
        
        # build objective function
        J = 0
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            J += ca.mtimes([(xk - xf).T, Q, (xk - xf)]) + ca.mtimes([uk.T, R, uk])

            # forward Euler
            x_next = xk + dt * f_plant(xk, uk)
            opti.subject_to(X[:, k + 1] == x_next)

            # collision avoidance with other agents' broadcast predictions
            for XYZ_m in XYZ_others:
                opti.subject_to(ca.sumsqr(X[0:3, k] - XYZ_m[:, k]) >= d_min ** 2)

        # terminal cost
        xN = X[:, N]
        J += ca.mtimes([(xN - xf).T, H, (xN - xf)])
        if term:
            opti.subject_to(xN == xf) # terminal constraint, xf

        # push initial interpolated predictions for warm-starting
        x0_m = x0_val[m, :].reshape(nx, 1)
        xf_m = xf_val[m, :].reshape(nx, 1)
        pred_X[m] = np.hstack([x0_m + (k / float(N)) * (xf_m - x0_m) for k in range(N + 1)])

        opti.minimize(J)
        opts = {
            "ipopt.print_level" : 0
        }
        opti.solver("ipopt", opts)
        return {"opti": opti, "X": X, "U": U, "x0": x0, "xf": xf, "XYZ_others": XYZ_others, "J" : J}

    # build agents and set goals
    agents = [build_agent_opti(m) for m in range(M)]

    # logs for plotting
    x_cl = np.zeros((M, nx, T + 1), dtype=float)
    x_cl[:, :, 0] = x0_val.copy()
    u_cl = np.zeros((M, nu, T), dtype=float)
    J_cl = np.zeros((M, T))
    wall_clk = np.zeros((M, T))

    Xt = x0_val.copy()

    # simulation loop
    for t in range(T):

        if mode == "jacobi":
            for m in range(M):
                set_xyz_others(m)

        for m in range(M):
            
            if mode == "gauss-seidel":
                set_xyz_others(m)
            
            # set initial-state parameters
            opti = agents[m]["opti"]
            X = agents[m]["X"]
            U = agents[m]["U"]
            J = agents[m]["J"]
            
            xt = Xt[m].reshape(nx, 1)
            opti.set_value(agents[m]["x0"], xt)
            opti.set_initial(X, pred_X[m]) # warm start
            opti.set_initial(U, pred_U[m]) # warm start
            
            sol = opti.solve()
            X_opt = sol.value(X)
            U_opt = sol.value(U)
            
            J_cl[m, t] = sol.value(J)
            wall_clk[m, t] = sol.stats()["t_wall_total"]
            
            pred_X[m] = shift_pred(X_opt)  # update shared predictions
            pred_U[m] = shift_pred(U_opt)  # update shared predictions


            ut = U_opt[:, 0].reshape((nu, 1))

            # apply first control, advance true states, shift warm starts, log
            if use_mj:
                xt_1 = f_true(m, mj_model, mj_data, ut)
                if vis_cfg is not None:
                    mj_vis_step(mj_data, vis_cfg)
            else:
                xt_1 = xt + dt * f_true(xt, ut)

            x_cl[m, :, t + 1] = xt_1.flatten()
            u_cl[m, :, t] = ut.flatten()
            
            Xt[m] = xt_1.flatten()
            


    # plot
    print("success, exiting...")

    J_avg = np.mean(J_cl)
    wall_clk = np.median(wall_clk)

    plot_t(env_cfg, x_cl, u_cl, J_avg, name, "decentralized", mode)
    plot_xyz(env_cfg, x_cl, J_avg, wall_clk, name, "decentralized", mode)
    animate_xyz_gif(env_cfg, x_cl, J_avg, wall_clk, name, "decentralized", mode)
    
    if vis_cfg is not None: 
        mj_cleanup(vis_cfg, name, "decentralized", mode)