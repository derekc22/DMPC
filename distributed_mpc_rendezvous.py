import numpy as np
import casadi as ca
from plot import *

def dmpc_distributed_rendezvous(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, f, f_np, sigma, obs, Q, R, H, term, dyn):

    t_max = T * dt
    shift = -1

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
        x0_shifted = np.roll(x0_val, shift=shift, axis=0)  # shift rows
        xf_val = x0_shifted
        opti.set_value(xf, xf_val.reshape((M * nx, 1)))
        
        # control bounds and initial condition constraint
        opti.subject_to(X[:, 0] == x0)
        for m in range(M):
            for i in range(nu):
                opti.subject_to(opti.bounded(U_lim[i][0], U[i, :], U_lim[i][1]))

            # obstacle constraint, center (xo,yo,zo), radius ro
            xz = X[nx * m : nx * (m + 1), :]
            for o in obs:
                xo, yo, zo, ro = o
                opti.subject_to((xz[0, :] - xo) ** 2 + (xz[1, :] - yo) ** 2 + (xz[2, :] - zo) ** 2 >= ro)
        
        # build objective function
        J = 0
        for k in range(N):
            for m in range(M):
                xk = X[nx * m : nx * (m + 1), k]
                uk = U[nu * m : nu * (m + 1), k]
                xf_m = xf[nx * m : nx * (m + 1)]

                J += ca.mtimes([(xk - xf_m).T, Q, (xk - xf_m)]) + ca.mtimes([uk.T, R, uk])

                # forward Euler
                x_next = xk + dt * f(xk, uk)
                opti.subject_to(X[nx * m : nx * (m + 1), k + 1] == x_next)

            # collision avoidance, pairwise
                for j in range(m + 1, M):
                    xz = X[nx * m : nx * m + 2, k]
                    xj = X[nx * j : nx * j + 2, k]
                    opti.subject_to(ca.sumsqr(xz - xj) >= d_min ** 2)

        # terminal cost
        for m in range(M):
            xN = X[nx * m:nx * (m + 1), N]
            xfN = xf[nx * m:nx * (m + 1), 0]
            J += ca.mtimes([(xN - xfN).T, H, (xN - xfN)])
            if term:
                opti.subject_to(xN == xf) # terminal constraint, xf
        
        # push initial interpolated predictions for warm-starting
        for m in range(M):
            x0_m = x0_val[m, :].reshape(nx, 1)
            xf_m = xf_val[m, :].reshape(nx, 1)
            pred_X[nx * m : nx * (m + 1), :] = np.hstack([x0_m + (k / float(N)) * (xf_m - x0_m) for k in range(N + 1)])
        
        opti.minimize(J)
        opts = {
            "ipopt.print_level" : 0
        }
        opti.solver("ipopt", opts)
        return {"opti": opti, "X": X, "U": U, "x0": x0, "xf": xf, "J" : J}

    planner = build_central_opti()
    
    # helpers
    def shift_pred(X):
        return np.hstack([X[:, 1:], X[:, -1:]])
    
    def set_xf_others(xt_val_others):
        xt_val = np.roll(xt_val_others, shift=shift, axis=0)  # shift rows
        planner["opti"].set_value(planner["xf"], xt_val.reshape(M * nx, 1))

    # logs for plotting
    x_cl = np.zeros((M, nx, T + 1), dtype=float)
    x_cl[:, :, 0] = x0_val.copy()
    u_cl = np.zeros((M, nu, T), dtype=float)
    J_cl = np.zeros((T))

    Xk = x0_val.copy()
    
    # store current position of others
    xt_val_others = x0_val

    # receding-horizon loop
    for k in range(T):

        # set initial-state parameters
        opti = planner["opti"]
        X = planner["X"]
        U = planner["U"]
        J = planner["J"]

        opti.set_value(planner["x0"], Xk.reshape(M * nx, 1))
        opti.set_initial(X, pred_X) # warm start
        opti.set_initial(U, pred_U) # warm start
        
        sol = opti.solve()
        X_opt = sol.value(X)
        U_opt = sol.value(U)
        
        pred_X = shift_pred(X_opt)  # update shared predictions
        pred_U = shift_pred(U_opt)  # update shared predictions

        set_xf_others(xt_val_others)

        for m in range(M):
            uk = U_opt[nu * m : nu * (m + 1), 0].reshape((nu, 1))

            # apply first control, advance true states, shift warm starts, log
            xk = Xk[m].reshape((nx, 1))
            xk_1 = xk + dt * f_np(xk, uk) #+ w[m][k, :].reshape(nx, 1)

            x_cl[m, :, k + 1] = xk_1.flatten()
            u_cl[m, :, k] = uk.flatten()
            
            Xk[m] = xk_1.flatten()
            
            xt_val_others = Xk
            
        J_cl[k] = sol.value(J)

            
    # plot
    J_cl_avg = np.mean(J_cl)/M
    plot_t(t_max, T, M, x_cl, u_cl, J_cl_avg, f"{dyn}_distributed_rendezvous")
    plot_xyz(M, x_cl, x0_val, None, J_cl_avg, obs, f"{dyn}_distributed_rendezvous")
    animate_xyz_gif(M, x_cl, x0_val, None, J_cl_avg, obs, f"{dyn}_distributed_rendezvous")