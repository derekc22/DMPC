import casadi as ca    
import numpy as np
from distributed_mpc import dmpc_distributed
from decentralized_mpc import dmpc_decentralized
from decentralized_mpc_client import dmpc_decentralized_client
from distributed_mpc_client import dmpc_distributed_client

# =========================================================================
# SETUP
# =========================================================================

# number of agents
M = 5

# minimum separation distance
d_min = 0.01

# discretization
dt = 0.5
N = 40
T = 50

# state and input dimensions
nx = 5
nu = 2

# states
# p:  3D position (global frame) - z is always 0
# θ:  heading angle (global frame)
# v:  forward velocity (global frame)

# inputs
# δ:  steering angle
# a:  longitudinal acceleration

# geometric
# wheelbase length
L = 1  # [m] 

# initial conditions
p0 = np.hstack([np.random.uniform(-10, 10, (M, 2)), np.zeros((M, 1))])
theta0 = np.zeros((M, 1))
v0 = np.zeros((M, 1))

# final conditions
pf = np.hstack([np.random.uniform(10, 11, (M, 2)), np.zeros((M, 1))])
thetaf = np.zeros((M, 1))
vf = np.zeros((M, 1))

# input bounds [δ, a]
delta_min = -0.7
delta_max = 0.7
a_min = -0.2
a_max = 0.2

# number of obstacles
n_obs = 0
p_obs = np.hstack([np.random.uniform(-10, 10, (n_obs, 2)), 5*np.ones((n_obs, 1))])
r_obs = np.random.uniform(1, 5, (n_obs, 1))
obs = np.hstack([p_obs, r_obs])

# cost matrices
Q = ca.DM(np.eye(nx))
R = ca.DM(np.eye(nu))
H = ca.DM(np.eye(nx))





# =========================================================================
# DERIVED VARIABLES
# =========================================================================

# derived quantities
x0_val = np.hstack([p0, theta0, v0])
xf_val = np.hstack([pf, thetaf, vf])
U_lim = [(delta_min, delta_max), (a_min, a_max)]





# =========================================================================
# DYNAMICS FUNCTIONS
# =========================================================================

# forward Euler integration
def f(x, u):
    theta, v = x[3], x[4]
    delta, a = u[0], u[1]
    x_dot = v * ca.cos(theta)
    y_dot = v * ca.sin(theta)
    z_dot = [0]
    theta_dot = (v / L) * delta
    v_dot = a
    return ca.vcat([x_dot, y_dot, z_dot, theta_dot, v_dot])

# non-CasADi forward Euler integration
def f_np(x, u):
    theta, v = x[3], x[4]
    delta, a = u[0], u[1]
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    z_dot = [0]
    theta_dot = (v / L) * delta
    v_dot = a
    return np.array([x_dot, y_dot, z_dot, theta_dot, v_dot])





# =========================================================================
# MPC CALLS
# =========================================================================

dmpc_decentralized(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "gauss-seidel", "bicycle")
dmpc_decentralized(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "jacobi", "bicycle")
dmpc_distributed(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "bicycle")
dmpc_decentralized_client(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val[0, :], f, f_np, 0, obs, Q, R, H, False, "gauss-seidel", "bicycle")
dmpc_decentralized_client(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val[0, :], f, f_np, 0, obs, Q, R, H, False, "jacobi", "bicycle")
dmpc_distributed_client(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val[0, :], f, f_np, 0, obs, Q, R, H, False, "bicycle")
