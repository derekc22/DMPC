import numpy as np
import casadi as ca
from utils.general_utils import cleanup
from utils.mj_utils import mj_vis_step, mj_step_state
from utils.mpc_utils import shift_pred, linear_warm_start, set_ipopt_solver, sphere_target, sphere_target_np, add_distributed_collision_constraints

def distributed_leader(dyn_cfg, dmpc_cfg, env_cfg, use_mj=False, vis_cfg=None):
    
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

    N, Q, R, H, term = (
        dmpc_cfg.N,
        dmpc_cfg.Q,
        dmpc_cfg.R,
        dmpc_cfg.H,
        dmpc_cfg.term,
    )

    T, dt, M, d_min, d_target, x0_val, obs, xf_val_leader = (
        env_cfg.T,
        env_cfg.dt,
        env_cfg.M,
        env_cfg.d_min,
        env_cfg.d_target,
        env_cfg.x0_val,
        env_cfg.obs,
        env_cfg.xf_val_leader,
    )

    def set_xf(xt_val_leader):
        xt_val = np.vstack([ xf_val_leader.reshape(nx, 1), np.tile(xt_val_leader.reshape(nx, 1), (M-1, 1)) ])
        planner["opti"].set_value(planner["xf"], xt_val)

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
                opti.subject_to(opti.bounded(u_lim[i][0], U[nu * m + i, :], u_lim[i][1]))

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
                    xk_target = ca.vertcat(sphere_target(xk[0:3], xf_m[0:3], d_target), xf_m[3:])
                    J += ca.mtimes([(xk - xk_target).T, Q, (xk - xk_target)]) + ca.mtimes([uk.T, R, uk])

                # forward Euler
                x_next = xk + dt * f_plant(xk, uk)
                opti.subject_to(X[nx * m : nx * (m + 1), k + 1] == x_next)

                # collision avoidance, pairwise
                if k > 0:
                    add_distributed_collision_constraints(opti, X, M, nx, d_min, k, m=m)

        # terminal collision avoidance, pairwise
        add_distributed_collision_constraints(opti, X, M, nx, d_min, N)

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
                xN_target = ca.vertcat(sphere_target(xN[0:3], xfN[0:3], d_target), xfN[3:])
                J += ca.mtimes([(xN - xN_target).T, H, (xN - xN_target)])
                if term:
                    # terminal constraint
                    opti.subject_to(ca.sumsqr(xN[0:3] - xfN[0:3]) == d_target ** 2) # be at target distance from leader
                    opti.subject_to(xN[3:] == xfN[3:])

        # push initial interpolated predictions for warm-starting
        x0_leader = x0_val_leader
        xf_leader = xf_val_leader.reshape(nx, 1)
        pred_X[nx * 0 : nx * (0 + 1), :] = linear_warm_start(x0_leader, xf_leader, N)
        
        for m in range(1, M):
            x0_m = x0_val[m, :].reshape(nx, 1)
            xf_m = sphere_target_np(x0_m, x0_val_leader, d_target)
            pred_X[nx * m : nx * (m + 1), :] = linear_warm_start(x0_m, xf_m, N)
        
        opti.minimize(J)
        set_ipopt_solver(opti)
        return {"opti": opti, "X": X, "U": U, "x0": x0, "xf": xf, "J" : J}

    planner = build_central_opti()
    
    # logs for plotting
    x_cl = np.zeros((M, nx, T + 1), dtype=float)
    x_cl[:, :, 0] = x0_val.copy()
    u_cl = np.zeros((M, nu, T), dtype=float)
    J_cl = np.zeros((T))
    wall_clk = np.zeros((T))

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
        
        J_cl[t] = sol.value(J)
        wall_clk[t] = sol.stats()["t_wall_total"]
        
        pred_X = shift_pred(X_opt)  # update shared predictions
        pred_U = shift_pred(U_opt)  # update shared predictions

        for m in range(M):
            ut = U_opt[nu * m : nu * (m + 1), 0].reshape((nu, 1))
            u_cl[m, :, t] = ut.flatten()

            # apply first control
            xt = Xt[m].reshape((nx, 1))
            if use_mj:
                f_true(m, mj_model, mj_data, ut)
            else:
                xt_1 = xt + dt * f_true(xt, ut)
                x_cl[m, :, t + 1] = xt_1.flatten()
                Xt[m] = xt_1.flatten()

        if use_mj:
            Xt = mj_step_state(mj_model, mj_data, M, nx)
            x_cl[:, :, t + 1] = Xt
            if vis_cfg is not None:
                mj_vis_step(mj_data, vis_cfg)

        xt_val_leader = Xt[0]
            

            
    # plot
    print("success, exiting...")
    cleanup(env_cfg, x_cl, u_cl, J_cl, wall_clk, name, "distributed_leader", vis_cfg, normalize_J_by_M=True)
