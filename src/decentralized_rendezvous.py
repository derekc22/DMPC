import numpy as np
import casadi as ca
from dataclasses import asdict
from utils.plot import *

def decentralized_rendezvous(dyn_cfg, dmpc_cfg, env_cfg, mj=False):
    
    # parse configs        
    if mj:
        dyn, f, f_np, nx, nu, U_lim, mj_model, mj_data = asdict(dyn_cfg).values()
    else:
        dyn, f, f_np, nx, nu, U_lim = asdict(dyn_cfg).values()    N, Q, R, H, term, mode = asdict(dmpc_cfg).values()
    T, dt, M, d_min, x0_val, obs, sigma = asdict(env_cfg).values()
    
    # helpers
    def shift_pred(X):
        return np.hstack([X[:, 1:], X[:, -1:]])

    def sphere_target(x_self, x_other, radius):
        # compute closest point on sphere of radius d_min around x_other to x_self
        diff = x_other - x_self
        dist_sq = ca.dot(diff, diff) + 1e-6  # squared distance with epsilon
        dist = ca.sqrt(dist_sq)  # smooth sqrt of non-zero value
        return x_other - radius * diff / dist

    def set_xyz_others(m):
        i = 0
        for j in range(M):
            if j == m:
                continue
            agents[m]["opti"].set_value(agents[m]["XYZ_others"][i], pred_X[j][0:3, :])
            i += 1
            
    def set_xf(xt_val_others):
        xt_val = np.roll(xt_val_others, shift=shift, axis=0)  # shift rows
        for m in range(M):
            agents[m]["opti"].set_value(agents[m]["xf"], xt_val[m])
    
    assert mode in ("gauss-seidel", "jacobi"), f"Invalid mode: {mode}"

    shift = -1

    # disturbances, per agent
    w = [np.random.multivariate_normal(np.zeros(nx), np.diag([sigma] * nx), T) for _ in range(M)]

    pred_X = np.zeros((M, nx, N + 1))
    pred_U = np.zeros((M, nu, N))

    # build a local OCP for one agent, with other agents' XYZ as parameters
    def build_agent_opti(m, xf_val):
        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        x0 = opti.parameter(nx, 1)
        xf = opti.parameter(nx, 1)

        # set final state
        opti.set_value(xf, xf_val.reshape((nx, 1)))
        
        # control bounds and initial condition constraint
        opti.subject_to(X[:, 0] == x0)
        for i in range(nu):
            opti.subject_to(opti.bounded(U_lim[i][0], U[i, :], U_lim[i][1]))

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
            
            # agents target sphere surface around other agents
            xk_target = ca.vertcat(sphere_target(xk[0:3], xf[0:3], d_min), xf[3:])
            J += ca.mtimes([(xk - xk_target).T, Q, (xk - xk_target)]) + ca.mtimes([uk.T, R, uk])

            # forward Euler
            x_next = xk + dt * f(xk, uk)
            opti.subject_to(X[:, k + 1] == x_next)

            # collision avoidance with other agents' broadcast predictions
            for XYZ_m in XYZ_others:
                opti.subject_to(ca.sumsqr(X[0:3, k] - XYZ_m[:, k]) >= d_min ** 2)

        # terminal cost
        # agents target sphere surface around other agents
        xN = X[:, N]
        xN_target = ca.vertcat(sphere_target(xN[0:3], xf[0:3], d_min), xf[3:])
        J += ca.mtimes([(xN - xN_target).T, H, (xN - xN_target)])
        if term:
            # terminal constraint
            opti.subject_to(ca.sumsqr(xN[0:3] - xf[0:3]) == d_min ** 2) # be at distance d_min from target agent
            opti.subject_to(xN[3:] == xf[3:])

        # push initial interpolated predictions for warm-starting
        x0_m = x0_val[m, :].reshape(nx, 1)
        xf_m = xf_val.reshape(nx, 1)
        pred_X[m] = np.hstack([x0_m + (k / float(N)) * (xf_m - x0_m) for k in range(N + 1)])

        opti.minimize(J)
        opts = {
            "ipopt.print_level" : 0
        }
        opti.solver("ipopt", opts)
        return {"opti": opti, "X": X, "U": U, "x0": x0, "xf": xf, "XYZ_others": XYZ_others, "J" : J}

    # build agents and set goals
    x0_shifted = np.roll(x0_val, shift=shift, axis=0)  # shift rows
    agents = [build_agent_opti(m, x0_shifted[m]) for m in range(M)]
    
    # logs for plotting
    x_cl = np.zeros((M, nx, T + 1), dtype=float)
    x_cl[:, :, 0] = x0_val.copy()
    u_cl = np.zeros((M, nu, T), dtype=float)
    J_cl = np.zeros((M, T))
    wall_clk = np.zeros((M, T))

    Xt = x0_val.copy()
    
    # store current position of others
    xt_val_others = x0_val

    # simulation loop
    for t in range(T):

        set_xf(xt_val_others)
        
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
            
            pred_X[m] = shift_pred(X_opt)  # update shared predictions
            pred_U[m] = shift_pred(U_opt)  # update shared predictions


            ut = U_opt[:, 0].reshape((nu, 1))

            # apply first control, advance true states, shift warm starts, log
            if mj:
                xt_1 = f_np(xt, ut, w[m][t, :], mj_model, mj_data)
            else:
                xt_1 = xt + dt * f_np(xt, ut) #+ w[m][t, :].reshape(nx, 1)

            x_cl[m, :, t + 1] = xt_1.flatten()
            u_cl[m, :, t] = ut.flatten()
            
            Xt[m] = xt_1.flatten()
            
            J_cl[m, t] = sol.value(J)
            
            xt_val_others = Xt

            wall_clk[m, t] = sol.stats()["t_wall_total"]


    # plot
    t_max = T * dt
    J_cl_avg = np.mean(J_cl)
    wall_clk_median = np.median(wall_clk)

    plot_t(t_max, T, M, x_cl, u_cl, J_cl_avg, dyn, "decentralized_rendezvous", mode)
    plot_xyz(M, x_cl, x0_val, None, J_cl_avg, obs, dyn, "decentralized_rendezvous", wall_clk_median, mode)
    animate_xyz_gif(M, x_cl, x0_val, None, J_cl_avg, obs, dyn, "decentralized_rendezvous", wall_clk_median, mode)