import numpy as np
import casadi as ca
from utils.plot import *

def dmpc_distributed_leader(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val_leader, f, f_np, sigma, obs, Q, R, H, term, dyn):

    wall_clk = np.zeros((T))

    # helpers
    def shift_pred(X):
        return np.hstack([X[:, 1:], X[:, -1:]])
    
    def sphere_target(x_self, x_other, radius):
        # compute closest point on sphere of radius d_min around x_other to x_self
        diff = x_other - x_self
        dist_sq = ca.dot(diff, diff) + 1e-6  # squared distance with epsilon
        dist = ca.sqrt(dist_sq)  # smooth sqrt of non-zero value
        return x_other - radius * diff / dist
    
    def set_xf(xt_val_leader):
        xt_val = np.vstack([ xf_val_leader.reshape(nx, 1), np.tile(xt_val_leader.reshape(nx, 1), (M-1, 1)) ])
        planner["opti"].set_value(planner["xf"], xt_val)

    # disturbances, per agent
    w = [np.random.multivariate_normal(np.zeros(nx), np.diag([sigma] * nx), T) for _ in range(M)]

    pred_X = np.zeros((M * nx, N + 1))
    pred_U = np.zeros((M * nu, N))

    # build a centralized OCP with stacked decision variables
    def build_central_opti():
        opti = ca.Opti()
        X = opti.variable(M * nx, N + 1)
        U = opti.variable(M * nu, N)
        x0 = opti.parameter(M * nx, 1)
        xf = opti.parameter(M * nx, 1)

        # set final states
        x0_val_leader = x0_val[0].reshape(nx, 1)
        xf_val = np.vstack([ xf_val_leader.reshape(nx, 1), np.tile(x0_val_leader, (M-1, 1)) ])
        opti.set_value(xf, xf_val)
        
        # control bounds and initial condition constraint
        opti.subject_to(X[:, 0] == x0)
        for m in range(M):
            for i in range(nu):
                opti.subject_to(opti.bounded(U_lim[i][0], U[nu * m + i, :], U_lim[i][1]))

            # obstacle constraint, center (xo,yo,zo), radius ro
            xz = X[nx * m : nx * (m + 1), :]
            for o in obs:
                xo, yo, zo, ro = o
                opti.subject_to((xz[0, :] - xo) ** 2 + (xz[1, :] - yo) ** 2 + (xz[2, :] - zo) ** 2 >= ro ** 2)
        
        # build objective function
        J = 0
        for k in range(N):
            for m in range(M):
                xk = X[nx * m : nx * (m + 1), k]
                uk = U[nu * m : nu * (m + 1), k]
                xf_m = xf[nx * m : nx * (m + 1)]

                if m == 0:
                    # leader targets exact goal position
                    J += ca.mtimes([(xk - xf_m).T, Q, (xk - xf_m)]) + ca.mtimes([uk.T, R, uk])
                else:
                    # followers target sphere surface around leader
                    xk_target = ca.vertcat(sphere_target(xk[0:3], xf_m[0:3], d_min), xf_m[3:])
                    J += ca.mtimes([(xk - xk_target).T, Q, (xk - xk_target)]) + ca.mtimes([uk.T, R, uk])

                # forward Euler
                x_next = xk + dt * f(xk, uk)
                opti.subject_to(X[nx * m : nx * (m + 1), k + 1] == x_next)

                # collision avoidance, pairwise
                for j in range(m + 1, M):
                    xz = X[nx * m : nx * m + 3, k]
                    xj = X[nx * j : nx * j + 3, k]
                    opti.subject_to(ca.sumsqr(xz - xj) >= d_min ** 2)

        # terminal cost
        for m in range(M):
            xN = X[nx * m:nx * (m + 1), N]
            xfN = xf[nx * m:nx * (m + 1), 0]
            if m == 0:
                # leader targets exact goal position
                J += ca.mtimes([(xN - xfN).T, H, (xN - xfN)])
                if term:
                    opti.subject_to(xN == xfN) # terminal constraint, xf
            else:
                # followers target sphere surface around leader
                xN_target = ca.vertcat(sphere_target(xN[0:3], xfN[0:3], d_min), xfN[3:])
                J += ca.mtimes([(xN - xN_target).T, H, (xN - xN_target)])
                if term:
                    # terminal constraint
                    opti.subject_to(ca.sumsqr(xN[0:3] - xfN[0:3]) == d_min ** 2) # be at distance d_min from leader
                    opti.subject_to(xN[3:] == xfN[3:])

        # push initial interpolated predictions for warm-starting
        x0_leader = x0_val_leader
        xf_leader = xf_val_leader.reshape(nx, 1)
        pred_X[nx * 0 : nx * (0 + 1), :] = np.hstack([x0_leader + (k / float(N)) * (xf_leader - x0_leader) for k in range(N + 1)])
        
        for m in range(1, M):
            x0_m = x0_val[m, :].reshape(nx, 1)
            xf_m = x0_val_leader
            pred_X[nx * m : nx * (m + 1), :] = np.hstack([x0_m + (k / float(N)) * (xf_m - x0_m) for k in range(N + 1)])
        
        opti.minimize(J)
        opts = {
            "ipopt.print_level" : 0
        }
        opti.solver("ipopt", opts)
        return {"opti": opti, "X": X, "U": U, "x0": x0, "xf": xf, "J" : J}

    planner = build_central_opti()
    
    # logs for plotting
    x_cl = np.zeros((M, nx, T + 1), dtype=float)
    x_cl[:, :, 0] = x0_val.copy()
    u_cl = np.zeros((M, nu, T), dtype=float)
    J_cl = np.zeros((T))

    Xt = x0_val.copy()
    x0_val_leader = x0_val[0, :] # store leader's current state
    
    # store current position of leader
    xt_val_leader = x0_val_leader

    # simulation loop
    for t in range(T):
        
        set_xf(xt_val_leader)

        # set initial-state parameters
        opti = planner["opti"]
        X = planner["X"]
        U = planner["U"]
        J = planner["J"]

        opti.set_value(planner["x0"], Xt.reshape(M * nx, 1))
        opti.set_initial(X, pred_X) # warm start
        opti.set_initial(U, pred_U) # warm start
        
        sol = opti.solve()
        X_opt = sol.value(X)
        U_opt = sol.value(U)
        
        pred_X = shift_pred(X_opt)  # update shared predictions
        pred_U = shift_pred(U_opt)  # update shared predictions

        for m in range(M):
            ut = U_opt[nu * m : nu * (m + 1), 0].reshape((nu, 1))

            # apply first control, advance true states, shift warm starts, log
            xt = Xt[m].reshape((nx, 1))
            xt_1 = xt + dt * f_np(xt, ut) #+ w[m][t, :].reshape(nx, 1)

            x_cl[m, :, t + 1] = xt_1.flatten()
            u_cl[m, :, t] = ut.flatten()
            
            Xt[m] = xt_1.flatten()
            
            if m == 0:
                xt_val_leader = Xt[0]
            
        J_cl[t] = sol.value(J)

        wall_clk[t] = sol.stats()["t_wall_total"]

            
    # plot
    t_max = T * dt
    J_cl_avg = np.mean(J_cl)/M
    wall_clk_median = np.median(wall_clk)

    plot_t(t_max, T, M, x_cl, u_cl, J_cl_avg, dyn, "distributed_leader")
    plot_xyz(M, x_cl, x0_val, xf_val_leader, J_cl_avg, obs, dyn, "distributed_leader", wall_clk_median)
    animate_xyz_gif(M, x_cl, x0_val, xf_val_leader, J_cl_avg, obs, dyn, "distributed_leader", wall_clk_median)