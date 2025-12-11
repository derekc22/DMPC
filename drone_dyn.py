import casadi as ca
import numpy as np
from distributed_mpc import dmpc_distributed
from decentralized_mpc import dmpc_decentralized
from decentralized_mpc_client import dmpc_decentralized_client
from distributed_mpc_client import dmpc_distributed_client

# =========================================================================
# HELPERS
# =========================================================================

def skew_np(v):
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ])

def skew_casadi(v):
    return ca.vertcat(
        ca.hcat([0,      -v[2],   v[1]]),
        ca.hcat([v[2],    0,     -v[0]]),
        ca.hcat([-v[1],   v[0],   0])
    )

def cross_casadi(a, b):
    return ca.vertcat(
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    )

def eul2rotm_zyx(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ])

    Ry = np.array([
        [ cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ])

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  cr, -sr],
        [0.0,  sr,  cr]
    ])

    return Rz @ Ry @ Rx





# =========================================================================
# SETUP
# =========================================================================

# number of agents
M = 5

# minimum separation distance
d_min = 0.01

# discretization
N = 40
dt = 0.05
T = 50

# state and input dimensions
nx = 12  
nu = 4

# states
# p:     3D position (global frame)
# euler: ZYX Euler angles - yaw, pitch, roll
# v:     3D linear velocity (global frame)
# wb:    3D angular velocity (body frame)

# inputs
# F:     upward thrust - F1, F2, F3, F4

# geometric
# arm length
d = 0.2  # [m] 

# dynamics & inertia
m = 0.5      # [kg]
Ixxb = 0.01  # [kg m^2]
Iyyb = 0.01  # [kg m^2]
Izzb = 0.05  # [kg m^2]

# initial conditions
p0 = np.hstack([np.random.uniform(-10, 10, (M, 2)), np.random.uniform(0, 10, (M, 1))]) # [m]
euler0 = np.zeros((M, 3))  # [rad]
v0 = np.zeros((M, 3))      # [m/s]
wb0 = np.zeros((M, 3))     # [rad/s]

# final conditions
pf = np.hstack([np.random.uniform(-10, 10, (M, 2)), np.random.uniform(0, 10, (M, 1))])
vf = np.zeros((M, 3))       # [rad]
eulerf = np.zeros((M, 3))   # [m/s]
wbf = np.zeros((M, 3))      # [rad/s]

# input bounds [F1, F2, F3, F4]
F_min = 0      # [N]
F_max = 20.0   # [N]

# number of obstacles
n_obs = 0
p_obs = np.hstack([np.random.uniform(-10, 10, (n_obs, 2)), 10*np.ones((n_obs, 1))])
r_obs = np.random.uniform(1, 5, (n_obs, 1))
obs = np.hstack([p_obs, r_obs])

# cost matrices
Q = ca.DM([
    50, 50, 50,         # position
    1e-3, 1e-3, 1e-3,   # euler angles
    1, 1, 1,            # linear velocity
    0.1, 0.1, 0.1       # angular velocity
])
Q = ca.diag(Q)
R = ca.DM(np.eye(nu))
H = 10.0 * Q





# =========================================================================
# DERIVED VARIABLES
# =========================================================================

x0_val = np.hstack([p0, euler0, v0, wb0])
xf_val = np.hstack([pf, eulerf, vf, wbf])

U_lim = [(F_min, F_max), (F_min, F_max), (F_min, F_max), (F_min, F_max)]

# inertia matrices
Ib_np = np.diag([Ixxb, Iyyb, Izzb])
Ib_inv_np = np.linalg.inv(Ib_np)

Ib = ca.diag(ca.DM([Ixxb, Iyyb, Izzb]))
Ib_inv = ca.inv(Ib)

# gravity (world frame)
g_val = 9.81
g_vec = np.array([0.0, 0.0, -g_val])

# rotor positions in body frame
r1_np = np.array([ d, 0.0, 0.0])
r2_np = np.array([ 0.0, d, 0.0])
r3_np = np.array([-d, 0.0, 0.0])
r4_np = np.array([ 0.0,-d, 0.0])

r1 = ca.DM(r1_np)
r2 = ca.DM(r2_np)
r3 = ca.DM(r3_np)
r4 = ca.DM(r4_np)





# =========================================================================
# DYNAMICS FUNCTIONS
# =========================================================================

def f(x, u):
    # x: (12,1), u: (4,1)  ->  dx: (12,1)

    # unpack
    yaw   = x[3]
    pitch = x[4]
    roll  = x[5]
    v     = x[6:9]      # 3x1
    wb    = x[9:12]     # 3x1
    p_b, q_b, r_b = wb[0], wb[1], wb[2]

    # rotor forces in body frame
    F1b = ca.vertcat(0, 0, u[0])
    F2b = ca.vertcat(0, 0, u[1])
    F3b = ca.vertcat(0, 0, u[2])
    F4b = ca.vertcat(0, 0, u[3])

    # rotation R = Rz(yaw) * Ry(pitch) * Rx(roll)
    cy, sy = ca.cos(yaw),   ca.sin(yaw)
    cp, sp = ca.cos(pitch), ca.sin(pitch)
    cr, sr = ca.cos(roll),  ca.sin(roll)

    Rz = ca.vertcat(
        ca.hcat([cy, -sy, 0]),
        ca.hcat([sy,  cy, 0]),
        ca.hcat([0,   0,  1]),
    )
    Ry = ca.vertcat(
        ca.hcat([ cp, 0, sp]),
        ca.hcat([ 0,  1, 0]),
        ca.hcat([-sp, 0, cp]),
    )
    Rx = ca.vertcat(
        ca.hcat([1, 0, 0]),
        ca.hcat([0, cr, -sr]),
        ca.hcat([0, sr,  cr]),
    )
    R = Rz @ Ry @ Rx

    # Euler-rate mapping ZYX
    tan_theta = ca.tan(pitch)
    sec_theta = 1.0 / ca.cos(pitch)
    roll_dot  = p_b + sr * tan_theta * q_b + cr * tan_theta * r_b
    pitch_dot = cr * q_b - sr * r_b
    yaw_dot   = sr * sec_theta * q_b + cr * sec_theta * r_b

    # cross for 3x1 columns
    def cross(a, b):
        return ca.vertcat(
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        )

    # torques from lever arms, no yaw drag torque
    tau = (cross(r1, F1b)
         + cross(r2, F2b)
         + cross(r3, F3b)
         + cross(r4, F4b)
         - cross(wb, Ib @ wb))

    # angular acceleration
    wb_dot = Ib_inv @ tau                      # 3x1

    # translational acceleration in world
    Fb_total = F1b + F2b + F3b + F4b           # 3x1
    F_world  = R @ Fb_total                    # 3x1
    g_col    = ca.DM([[0.0], [0.0], [-g_val]]) # 3x1
    v_dot    = F_world / m + g_col             # 3x1

    # state derivative, order matches the state
    dx = ca.vertcat(
        v,
        ca.vertcat(yaw_dot, pitch_dot, roll_dot),
        v_dot,
        wb_dot,
    )
    return dx

def f_np(x, u):
    # x: (12,1), u: (4,1)  ->  dx: (12,1)

    # unpack using
    yaw   = x[3][0]
    pitch = x[4][0]
    roll  = x[5][0]
    v     = x[6:9]          # (3,1)
    wb    = x[9:12]         # (3,1)
    p_b, q_b, r_b = wb[0][0], wb[1][0], wb[2][0]

    # rotor forces in body frame, 3x1 columns
    F1b = np.array([[0.0], [0.0], [u[0, 0]]])
    F2b = np.array([[0.0], [0.0], [u[1, 0]]])
    F3b = np.array([[0.0], [0.0], [u[2, 0]]])
    F4b = np.array([[0.0], [0.0], [u[3, 0]]])

    # rotation R = Rz(yaw) * Ry(pitch) * Rx(roll)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]])
    R = Rz @ Ry @ Rx

    # Euler-rate mapping ZYX
    tan_theta = np.tan(pitch)
    sec_theta = 1.0 / np.cos(pitch)
    roll_dot  = p_b + sr * tan_theta * q_b + cr * tan_theta * r_b
    pitch_dot = cr * q_b - sr * r_b
    yaw_dot   = sr * sec_theta * q_b + cr * sec_theta * r_b

    # cross for 3x1 columns
    def cross(a, b):
        ax, ay, az = a[0, 0], a[1, 0], a[2, 0]
        bx, by, bz = b[0, 0], b[1, 0], b[2, 0]
        return np.array([[ay*bz - az*by],
                         [az*bx - ax*bz],
                         [ax*by - ay*bx]])

    # arm vectors as 3x1 columns
    r1c = np.array([[ r1_np[0]], [ r1_np[1]], [ r1_np[2]]])
    r2c = np.array([[ r2_np[0]], [ r2_np[1]], [ r2_np[2]]])
    r3c = np.array([[ r3_np[0]], [ r3_np[1]], [ r3_np[2]]])
    r4c = np.array([[ r4_np[0]], [ r4_np[1]], [ r4_np[2]]])

    # torques from lever arms, no yaw drag torque
    tau = (cross(r1c, F1b)
         + cross(r2c, F2b)
         + cross(r3c, F3b)
         + cross(r4c, F4b)
         - cross(wb, Ib_np @ wb))

    # angular acceleration
    wb_dot = Ib_inv_np @ tau                 # (3,1)

    # translational acceleration in world
    Fb_total = F1b + F2b + F3b + F4b         # (3,1)
    F_world  = R @ Fb_total                  # (3,1)
    g_col    = np.array([[0.0], [0.0], [-g_val]])
    v_dot    = F_world / m + g_col           # (3,1)

    # state derivative, order matches the state
    dx = np.vstack([
        v,
        np.array([[yaw_dot], [pitch_dot], [roll_dot]]),
        v_dot,
        wb_dot,
    ])
    return dx  # (12,1)





# =========================================================================
# MPC CALLS
# =========================================================================

dmpc_decentralized(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "gauss-seidel", "drone")
dmpc_decentralized(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "jacobi", "drone")
dmpc_distributed(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "drone")
dmpc_decentralized_client(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val[0, :], f, f_np, 0, obs, Q, R, H, False, "gauss-seidel", "drone")
dmpc_decentralized_client(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val[0, :], f, f_np, 0, obs, Q, R, H, False, "jacobi", "drone")
dmpc_distributed_client(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val[0, :], f, f_np, 0, obs, Q, R, H, False, "drone")