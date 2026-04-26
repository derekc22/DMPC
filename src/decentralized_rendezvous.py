import numpy as np
import casadi as ca
from utils.general_utils import cleanup
from utils.mj_utils import mj_vis_step, mj_step_state
from utils.mpc_utils import shift_pred, linear_warm_start, set_ipopt_solver, sphere_target, sphere_target_np, set_xyz_others, add_decentralized_collision_constraints

def decentralized_rendezvous(dyn_cfg, dmpc_cfg, env_cfg, use_mj=False, vis_cfg=None):
    
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

    T, dt, M, d_min, d_target, x0_val, obs = (
        env_cfg.T,
        env_cfg.dt,
        env_cfg.M,
        env_cfg.d_min,
        env_cfg.d_target,
        env_cfg.x0_val,
        env_cfg.obs,
    )
    
    def set_xf(xt_val_others):
        xt_val = np.roll(xt_val_others, shift=shift, axis=0)  # shift rows
        for m in range(M):
            agents[m]["opti"].set_value(agents[m]["xf"], xt_val[m])
    
    assert mode in ("gauss-seidel", "jacobi"), f"Invalid mode: {mode}"

    shift = -1

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
            
            # agents target sphere surface around other agents
            xk_target = ca.vertcat(sphere_target(xk[0:3], xf[0:3], d_target), xf[3:])
            J += ca.mtimes([(xk - xk_target).T, Q, (xk - xk_target)]) + ca.mtimes([uk.T, R, uk])

            # forward Euler
            x_next = xk + dt * f_plant(xk, uk)
            opti.subject_to(X[:, k + 1] == x_next)

            # collision avoidance with other agents' broadcast predictions
            if k > 0:
                add_decentralized_collision_constraints(opti, X, XYZ_others, d_min, k)

        # terminal collision avoidance with other agents' broadcast predictions
        add_decentralized_collision_constraints(opti, X, XYZ_others, d_min, N)

        # terminal cost
        # agents target sphere surface around other agents
        xN = X[:, N]
        xN_target = ca.vertcat(sphere_target(xN[0:3], xf[0:3], d_target), xf[3:])
        J += ca.mtimes([(xN - xN_target).T, H, (xN - xN_target)])
        if term:
            # terminal constraint
            opti.subject_to(ca.sumsqr(xN[0:3] - xf[0:3]) == d_target ** 2) # be at target distance from target agent
            opti.subject_to(xN[3:] == xf[3:])

        # push initial interpolated predictions for warm-starting
        x0_m = x0_val[m, :].reshape(nx, 1)
        xf_m = sphere_target_np(x0_m, xf_val, d_target)
        pred_X[m] = linear_warm_start(x0_m, xf_m, N)

        opti.minimize(J)
        set_ipopt_solver(opti)
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
                set_xyz_others(agents, pred_X, M, m)

        for m in range(M):
            
            if mode == "gauss-seidel":
                set_xyz_others(agents, pred_X, M, m)
                
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
            u_cl[m, :, t] = ut.flatten()

            # apply first control
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

        xt_val_others = Xt



    # plot
    print("success, exiting...")
    cleanup(env_cfg, x_cl, u_cl, J_cl, wall_clk, name, "decentralized_rendezvous", vis_cfg, mode)
