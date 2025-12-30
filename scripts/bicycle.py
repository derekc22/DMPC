import casadi as ca    
import numpy as np
from src.distributed import distributed
from src.decentralized import decentralized
from src.decentralized_leader import decentralized_leader
from src.distributed_leader import distributed_leader
from src.distributed_rendezvous import distributed_rendezvous
from src.decentralized_rendezvous import decentralized_rendezvous
from config.dmpc_cfg import DistributedParams, DecentralizedParams
from config.dyn_cfg import DynamicsParams
from config.env_cfg import EnvParams, LeaderEnvParams, RendezvousEnvParams

# =========================================================================
# SETUP
# =========================================================================

# number of agents
M = 2

# minimum separation distance
d_min = 0.1

# discretization
dt = 0.5
N = 25
T = 100

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
pf = np.hstack([np.random.uniform(-10, 10, (M, 2)), np.zeros((M, 1))])
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
u_lim = [(delta_min, delta_max), (a_min, a_max)]





# =========================================================================
# DYNAMICS FUNCTIONS
# =========================================================================

# forward Euler integration
def f_plant(x, u):
    theta, v = x[3], x[4]
    delta, a = u[0], u[1]
    x_dot = v * ca.cos(theta)
    y_dot = v * ca.sin(theta)
    z_dot = [0]
    theta_dot = (v / L) * delta
    v_dot = a
    return ca.vcat([x_dot, y_dot, z_dot, theta_dot, v_dot])

# non-CasADi forward Euler integration
def f_true(x, u):
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
dyn_np_cfg = DynamicsParams(name="bicycle", f_plant=f_plant, f_true=f_true, nx=nx, nu=nu, u_lim=u_lim)

decentr_cfg_gauss = DecentralizedParams(N=N, Q=Q, R=R, H=H, term=False, mode="gauss-seidel")
decentr_cfg_jacbi = DecentralizedParams(N=N, Q=Q, R=R, H=H, term=False, mode="jacobi")
distr_cfg = DistributedParams(N=N, Q=Q, R=R, H=H, term=False)

rndzvs_env_cfg = RendezvousEnvParams(T=T, dt=dt, M=M, d_min=d_min, x0_val=x0_val, obs=obs)
ldr_env_cfg = LeaderEnvParams(T=T, dt=dt, M=M, d_min=d_min, x0_val=x0_val, xf_val_leader=xf_val[0, :], obs=obs)
env_cfg = EnvParams(T=T, dt=dt, M=M, d_min=d_min, x0_val=x0_val, xf_val=xf_val, obs=obs)


decentralized_rendezvous(dyn_np_cfg, decentr_cfg_gauss, rndzvs_env_cfg)
decentralized_rendezvous(dyn_np_cfg, decentr_cfg_jacbi, rndzvs_env_cfg)
distributed_rendezvous(dyn_np_cfg, distr_cfg, rndzvs_env_cfg)

decentralized_leader(dyn_np_cfg, decentr_cfg_gauss, ldr_env_cfg)
decentralized_leader(dyn_np_cfg, decentr_cfg_jacbi, ldr_env_cfg)
distributed_leader(dyn_np_cfg, distr_cfg, ldr_env_cfg)

decentralized(dyn_np_cfg, decentr_cfg_gauss, env_cfg)
decentralized(dyn_np_cfg, decentr_cfg_jacbi, env_cfg)
distributed(dyn_np_cfg, distr_cfg, env_cfg)