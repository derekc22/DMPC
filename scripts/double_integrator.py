import casadi as ca    
import numpy as np
from src.distributed_mpc import dmpc_distributed
from src.decentralized_mpc import dmpc_decentralized
from src.decentralized_mpc_leader import dmpc_decentralized_leader
from src.distributed_mpc_leader import dmpc_distributed_leader
from src.distributed_mpc_rendezvous import dmpc_distributed_rendezvous
from src.decentralized_mpc_rendezvous import dmpc_decentralized_rendezvous

# =========================================================================
# SETUP
# =========================================================================

# number of agents
M = 3

# minimum separation distance
d_min = 0.1

# discretization
dt = 0.1
N = 25
T = 100

# state and input dimensions
nx = 12
nu = 6

# inertial
mass = 50
I = np.diag([10, 10, 10])

# initial conditions
p0 = np.random.randint(-50, 50, (M, 3))
theta0 = np.zeros((M, 3))
v0 = np.zeros((M, 3))
theta_dot0 = np.zeros((M, 3))

# final conditions
pf = np.random.randint(-10, 10, (M, 3))
thetaf = np.zeros((M, 3))
vf = np.zeros((M, 3))
theta_dotf = np.zeros((M, 3))

# input bounds [F, tau]
force_min = -100
force_max = 100
torque_min = -50
torque_max = 50

# number of obstacles
n_obs = 0
p_obs = np.hstack([np.random.uniform(-10, 10, (n_obs, 2)), 5*np.ones((n_obs, 1))])
r_obs = np.random.uniform(1, 5, (n_obs, 1))
obs = np.hstack([p_obs, r_obs])

# cost matrices
Q = 10*ca.DM(np.eye(nx))
R = 0.01*ca.DM(np.eye(nu))
H = 10*ca.DM(np.eye(nx))






# =========================================================================
# DERIVED VARIABLES
# =========================================================================

# derived quantities
x0_val = np.hstack([p0, theta0, v0, theta_dot0])
xf_val = np.hstack([pf, thetaf, vf, theta_dotf])
U_lim = [(force_min, force_max)] * 3 + [(torque_min, torque_max)] * 3





# =========================================================================
# DYNAMICS FUNCTIONS
# =========================================================================

# forward Euler integration
def f(x, u):
    x_dot = x[6]
    y_dot = x[7]
    z_dot = x[8]
    theta_x_dot = x[9]
    theta_y_dot = x[10]
    theta_z_dot = x[11]
    
    Fx = u[0] 
    Fy = u[1] 
    Fz = u[2] 
    tau_x = u[3] 
    tau_y = u[4] 
    tau_z = u[5] 
    
    x_ddot = (Fx/mass)
    y_ddot = (Fy/mass)
    z_ddot = (Fz/mass)
    
    theta_x_ddot = (tau_x/I[0, 0])
    theta_y_ddot = (tau_y/I[1, 1])
    theta_z_ddot = (tau_z/I[2, 2])
    
    return ca.vcat([x_dot, y_dot, z_dot, 
                    theta_x_dot, theta_y_dot, theta_z_dot,
                    x_ddot, y_ddot, z_ddot, 
                    theta_x_ddot, theta_y_ddot, theta_z_ddot
                    ])

# non-CasADi forward Euler integration
def f_np(x, u):
    x_dot, y_dot, z_dot, theta_x_dot, theta_y_dot, theta_z_dot = x[6:]
    Fx, Fy, Fz, tau_x, tau_y, tau_z = u
    
    x_ddot = (Fx/mass)
    y_ddot = (Fy/mass)
    z_ddot = (Fz/mass)
    
    theta_x_ddot = (tau_x/I[0, 0])
    theta_y_ddot = (tau_y/I[1, 1])
    theta_z_ddot = (tau_z/I[2, 2])
    
    return np.array([x_dot, y_dot, z_dot, 
                     theta_x_dot, theta_y_dot, theta_z_dot,
                     x_ddot, y_ddot, z_ddot, 
                     theta_x_ddot, theta_y_ddot, theta_z_ddot
                     ])





# =========================================================================
# MPC CALLS
# =========================================================================

dmpc_decentralized_rendezvous(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, f, f_np, 0, obs, Q, R, H, False, "gauss-seidel", "double_integrator")
dmpc_decentralized_rendezvous(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, f, f_np, 0, obs, Q, R, H, False, "jacobi", "double_integrator")
dmpc_distributed_rendezvous(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, f, f_np, 0, obs, Q, R, H, False, "double_integrator")

dmpc_decentralized_leader(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val[0, :], f, f_np, 0, obs, Q, R, H, False, "gauss-seidel", "double_integrator")
dmpc_decentralized_leader(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val[0, :], f, f_np, 0, obs, Q, R, H, False, "jacobi", "double_integrator")
dmpc_distributed_leader(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val[0, :], f, f_np, 0, obs, Q, R, H, False, "double_integrator")

dmpc_decentralized(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "gauss-seidel", "double_integrator")
dmpc_decentralized(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "jacobi", "double_integrator")
dmpc_distributed(T, M, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "double_integrator")