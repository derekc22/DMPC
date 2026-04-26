import casadi as ca
import numpy as np


def shift_pred(X):
    return np.hstack([X[:, 1:], X[:, -1:]])


def linear_warm_start(x0, xf, N):
    return np.hstack([x0 + (k / float(N)) * (xf - x0) for k in range(N + 1)])


def set_ipopt_solver(opti):
    opts = {
        "ipopt.print_level" : 0
    }
    opti.solver("ipopt", opts)


def sphere_target(x_self, x_other, radius):
    diff = x_other - x_self
    dist = ca.sqrt(ca.dot(diff, diff) + 1e-6)
    return x_other - radius * diff / dist


def sphere_target_np(x_self, x_other, radius):
    x_self = np.asarray(x_self, dtype=float).reshape(-1)
    x_other = np.asarray(x_other, dtype=float).reshape(-1)
    diff = x_other[0:3] - x_self[0:3]
    dist = np.sqrt(np.dot(diff, diff) + 1e-6)
    x_target = x_other.copy()
    x_target[0:3] = x_other[0:3] - radius * diff / dist
    return x_target.reshape(-1, 1)


def set_xyz_others(agents, pred_X, M, m):
    i = 0
    for j in range(M):
        if j == m:
            continue
        agents[m]["opti"].set_value(agents[m]["XYZ_others"][i], pred_X[j][0:3, :])
        i += 1


def add_decentralized_collision_constraints(opti, X, XYZ_others, d_min, k):
    for XYZ_m in XYZ_others:
        opti.subject_to(ca.sumsqr(X[0:3, k] - XYZ_m[:, k]) >= d_min ** 2)


def add_distributed_collision_constraints(opti, X, M, nx, d_min, k, m=None):
    m_iter = range(M) if m is None else range(m, m + 1)
    for i in m_iter:
        for j in range(i + 1, M):
            xz = X[nx * i : nx * i + 3, k]
            xj = X[nx * j : nx * j + 3, k]
            opti.subject_to(ca.sumsqr(xz - xj) >= d_min ** 2)
