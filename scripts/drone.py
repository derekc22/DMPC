import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from src.distributed import distributed
from src.decentralized import decentralized
from src.decentralized_leader import decentralized_leader
from src.distributed_leader import distributed_leader
from src.distributed_rendezvous import distributed_rendezvous
from src.decentralized_rendezvous import decentralized_rendezvous
from config.dmpc_cfg import DistributedParams, DecentralizedParams
from config.dyn_cfg import MuJoCoDynamicsParams
from config.env_cfg import EnvParams, LeaderEnvParams, RendezvousEnvParams
from config.xml_cfg import XMLParams
from config.vis_cfg import VisualizationParams
from utils.mj_utils import generate_xml, load_model, reset_model, init_vis

# =========================================================================
# SETUP
# =========================================================================

# number of agents
M = 3

# minimum separation distance
d_min = 0.0

# discretization
dt = 0.01
N = 400
T = 150

# state and input dimensions
nx = 6
nu = 3

nq = 7
nv = 6
na = 4

# geometric
# arm length
d = 0.2  # [m] 

# dynamics & inertia
mass = 0.5      # [kg]
Ixxb = 0.01  # [kg m^2]
Iyyb = 0.01  # [kg m^2]
Izzb = 0.005  # [kg m^2]

# initial conditions
p0_true = np.hstack([np.random.uniform(-5, 5, (M, 2)), np.random.uniform(1, 5, (M, 1))]) # [m]
quat0_true = np.tile([1.0, 0.0, 0.0, 0.0], (M, 1))  # [w, x, y, z] identity quaternion
v0_true = np.zeros((M, 3))      # [m/s]
wb0_true = np.zeros((M, 3))     # [rad/s]

p0 = p0_true.copy()
v0 = v0_true.copy()

# final conditions
pf = np.hstack([np.random.uniform(-5, 5, (M, 2)), np.random.uniform(5, 10, (M, 1))])
vf = np.zeros((M, 3))   # [m/s]

# input bounds [fx, fy, fz]
f_min_x = -1.00   # [N]
f_max_x =  1.00   # [N]
f_min_y = -1.00   # [N]
f_max_y =  1.00   # [N]
f_min_z =  0.00   # [N]
f_max_z =  5.00   # [N]

# Rotor thrust limits for allocation/clipping (these are actuator-space limits).
# Kept local to avoid changing non-dynamics parts of the file.
rotor_f_min = 0.0
rotor_f_max = 30.0

# number of obstacles
n_obs = 0
p_obs = np.hstack([np.random.uniform(-5, 5, (n_obs, 2)), np.random.uniform(0, 5, (n_obs, 1))])
r_obs = np.random.uniform(0.1, 0.5, (n_obs, 1))
obs = np.hstack([p_obs, r_obs])

# cost matrices
Q = ca.DM([
    5, 5, 5,     # position
    1, 1, 1,     # linear velocity
])
Q = ca.diag(Q)
R = 2*ca.DM(np.eye(nu))
H = 10.0 * Q





# =========================================================================
# DERIVED VARIABLES
# =========================================================================

q0_val = np.hstack([p0_true, quat0_true, v0_true, wb0_true])
x0_val = np.hstack([p0, v0])
xf_val= np.hstack([pf, vf])

u_lim = [(f_min_x, f_max_x), (f_min_y, f_max_y), (f_min_z, f_max_z)]

# inertia matrices
Ib_np = np.diag([Ixxb, Iyyb, Izzb])
Ib_mnv_np = np.linalg.inv(Ib_np)

Ib = ca.diag(ca.DM([Ixxb, Iyyb, Izzb]))
Ib_mnv = ca.inv(Ib)

# gravity (world frame)
g_val = 9.81
g_vec = np.array([0.0, 0.0, -g_val])

# rotor positions in body frame
r1 = ca.DM([ d, 0.0, 0.0])
r2 = ca.DM([ 0.0, d, 0.0])
r3 = ca.DM([-d, 0.0, 0.0])
r4 = ca.DM([ 0.0,-d, 0.0])





# =========================================================================
# DYNAMICS FUNCTIONS
# =========================================================================

def f_plant(x, u):
    # x: (6,1), u: (3,1)  ->  dx: (6,1)
    # unpack
    v     = x[3:]     # 3x1
    
    g_col    = ca.DM([[0.0], [0.0], [-g_val]]) # 3x1

    # state derivative, order matches the state
    dx = ca.vertcat(
        v,
        u/mass + g_col,
    )
    return dx

def quat_to_rotmat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert mujoco quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    quat_wxyz = np.asarray(quat_wxyz, dtype=float).reshape(4,)
    mat9 = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(mat9, quat_wxyz)
    return mat9.reshape(3, 3)


def vee_so3(S: np.ndarray) -> np.ndarray:
    """vee operator for a 3x3 skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def desired_rotation_from_fnet(Fnet_w: np.ndarray, psi: float) -> np.ndarray:
    """Construct desired body-to-world rotation R_d given net force direction and yaw."""
    Fnet_w = np.asarray(Fnet_w, dtype=float).reshape(3,)
    nF = np.linalg.norm(Fnet_w)
    if nF < 1e-8:
        # Degenerate, fall back to level attitude.
        return np.eye(3)

    b3d = Fnet_w / nF

    b1psi = np.array([np.cos(psi), np.sin(psi), 0.0], dtype=float)
    # Project b1psi onto plane orthogonal to b3d
    b1 = b1psi - np.dot(b1psi, b3d) * b3d
    nb1 = np.linalg.norm(b1)
    if nb1 < 1e-8:
        # Yaw singularity when b3d is parallel to b1psi, pick an arbitrary orthogonal axis.
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
        b1 = tmp - np.dot(tmp, b3d) * b3d
        nb1 = np.linalg.norm(b1)
        if nb1 < 1e-8:
            tmp = np.array([0.0, 1.0, 0.0], dtype=float)
            b1 = tmp - np.dot(tmp, b3d) * b3d
            nb1 = np.linalg.norm(b1)
    b1d = b1 / nb1

    b2d = np.cross(b3d, b1d)
    b2d = b2d / max(np.linalg.norm(b2d), 1e-12)

    Rd = np.column_stack([b1d, b2d, b3d])
    return Rd


def attitude_pd_so3(R: np.ndarray,
                    omega_b: np.ndarray,
                    Rd: np.ndarray,
                    J: np.ndarray,
                    kR: float,
                    kW: float) -> np.ndarray:
    """Geometric PD attitude controller on SO(3). Returns body torque tau (3,)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    Rd = np.asarray(Rd, dtype=float).reshape(3, 3)
    omega_b = np.asarray(omega_b, dtype=float).reshape(3,)
    J = np.asarray(J, dtype=float).reshape(3, 3)

    eR_mat = Rd.T @ R - R.T @ Rd
    eR = 0.5 * vee_so3(eR_mat)

    # Desired body rates are zero in this minimal implementation.
    eW = omega_b

    tau = -kR * eR - kW * eW + np.cross(omega_b, J @ omega_b)

    # This mujoco actuator model has no rotor drag torque, so yaw authority is absent.
    tau[2] = 0.0
    return tau


def allocate_thrust_torque_to_rotors(T: float,
                                    tau_xy: np.ndarray,
                                    arm: float,
                                    f_min: float,
                                    f_max: float) -> np.ndarray:
    """Allocate total thrust and roll/pitch torque into 4 rotor thrusts.

    Rotor order matches the actuator XML:
      1) green  at (0, +d)
      2) red    at (+d, 0)
      3) blue   at (-d, 0)
      4) black  at (0, -d)

    With forces along body +z, the moment from rotor i is r_i x (f_i e3):
      tau_x = d (f_green - f_black)
      tau_y = d (f_blue  - f_red)

    There is no yaw torque term in this actuator model.
    """
    tau_xy = np.asarray(tau_xy, dtype=float).reshape(2,)
    tau_x, tau_y = float(tau_xy[0]), float(tau_xy[1])

    A = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [arm, 0.0, 0.0, -arm],
        [0.0, -arm, arm, 0.0],
    ], dtype=float)
    b = np.array([float(T), tau_x, tau_y], dtype=float)

    # Minimum-norm solution
    f = A.T @ np.linalg.solve(A @ A.T, b)

    f = np.clip(f, f_min, f_max)
    return f


def f_true(m, mj_model, mj_data, mu):
    """True mujoco step with a low-level geometric controller.

    IMPORTANT CONSISTENCY:
    The reduced model used by MPC is:
        p_dot = v
        v_dot = u/mass + g
    Therefore, the MPC input u (here: mu) corresponds to the WORLD-frame thrust force
    (i.e., the force produced by the rotors along body b3, expressed in world frame).
    Gravity is already handled by +g in the reduced model and by mujoco, so we DO NOT
    add mass*g again here.

    Mapping:
      F_des_w = mu
      R_d     from (F_des_w, yaw)
      T       = F_des_w^T (R e3)   (project desired force onto current thrust axis)
      tau     = SO(3) PD torque (roll/pitch only)
      f_i     rotor thrust commands
    """
    # --- parameters ---
    arm = float(d)    # rotor arm length used in XML site positions
    kR = 6.0
    kW = 0.8
    psi_des = 0.0


    # Desired WORLD-frame thrust force from MPC
    Fdes_w = np.asarray(mu, dtype=float).reshape(3,)

    qpos = mj_data.qpos.reshape(M, nq)[m]
    qvel = mj_data.qvel.reshape(M, nv)[m]

    quat_wxyz = qpos[3:7]
    R_bw = quat_to_rotmat_wxyz(quat_wxyz)  # body-to-world
    omega = qvel[3:6]

    Rd = desired_rotation_from_fnet(Fdes_w, psi_des)

    e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    b3 = R_bw @ e3

    # Total thrust magnitude along current b3 needed to realize the requested force
    T = float(np.dot(Fdes_w, b3))
    T = float(np.clip(T, 0.0, 4.0 * rotor_f_max))

    tau = attitude_pd_so3(R_bw, omega, Rd, Ib_np, kR, kW)

    rotor_f = allocate_thrust_torque_to_rotors(T, tau[:2], arm, rotor_f_min, rotor_f_max)

    ctrl_start = int(m) * int(na)
    mj_data.ctrl[ctrl_start:ctrl_start + na] = rotor_f
    mujoco.mj_step(mj_model, mj_data)

    xpos = mj_data.qpos.reshape(M, nq)[m]
    xvel = mj_data.qvel.reshape(M, nv)[m]

    x = np.hstack([xpos[:3], xvel[:3]])
    return x  # (6,)





# =========================================================================
# MUJOCO INITIALIZATION
# =========================================================================

agent_xml = f"""                
            <body name="agent_m" pos="0 0 0">
                <inertial pos="0 0 0" mass="{mass}" diaginertia="{Ixxb} {Iyyb} {Izzb}"/> 
                <joint type="free"/>

                <geom name="green_arm_m" type="box" pos="     0   {d/2}  0" size="0.01  {d/2}  0.01" rgba="0 1 0 1" />
                <geom name="red_arm_m"   type="box" pos=" {d/2}       0  0" size="{d/2}  0.01  0.01" rgba="1 0 0 1" />
                <geom name="blue_arm_m"  type="box" pos="-{d/2}       0  0" size="{d/2}  0.01  0.01" rgba="0 0 1 1" />
                <geom name="black_arm_m" type="box" pos="     0  -{d/2}  0" size="0.01  {d/2}  0.01" rgba="0 0 0 1" />

                <site name="green_prop_m" type="box" size=".01 .01 .02" pos="    0   {d}  0.01" quat="1 0 0 0" rgba="1 1 1 1" />
                <site name="red_prop_m"   type="box" size=".01 .01 .02" pos="  {d}     0  0.01" quat="1 0 0 0" rgba="1 1 1 1" />
                <site name="blue_prop_m"  type="box" size=".01 .01 .02" pos=" -{d}     0  0.01" quat="1 0 0 0" rgba="1 1 1 1" />
                <site name="black_prop_m" type="box" size=".01 .01 .02" pos="    0  -{d}  0.01" quat="1 0 0 0" rgba="1 1 1 1" />
            </body>
"""

actuator_xml = f"""
            <general site="green_prop_m"  gear="0 0 1 0 0 0"  ctrlrange="{rotor_f_min} {rotor_f_max}"  ctrllimited="true"/>
            <general site="red_prop_m"    gear="0 0 1 0 0 0"  ctrlrange="{rotor_f_min} {rotor_f_max}"  ctrllimited="true"/>
            <general site="blue_prop_m"   gear="0 0 1 0 0 0"  ctrlrange="{rotor_f_min} {rotor_f_max}"  ctrllimited="true"/>
            <general site="black_prop_m"  gear="0 0 1 0 0 0"  ctrlrange="{rotor_f_min} {rotor_f_max}"  ctrllimited="true"/>
"""

xml_cfg = XMLParams(name="drone", agent_xml=agent_xml, actuator_xml=actuator_xml, gravity=g_val, dt=dt, M=M, nq=nq, q0_val=q0_val, obs=obs)
xml = generate_xml(xml_cfg)

mj_model, mj_data = load_model(xml)
reset_model(mj_model, mj_data)

presets = {
    "distance": 2, 
    "azimuth": 0, 
    "elevation": -30
}
vis_cfg = VisualizationParams(presets=presets, 
                              track=True, 
                              show_world_csys=True, 
                              show_body_csys=True, 
                              vid_width=1280, 
                              vid_height=720,
                              vid_fps=30.0,
                              enable_viewer=False)

init_vis(mj_model, mj_data, vis_cfg=vis_cfg)





# =========================================================================
# CONFIGS
# =========================================================================
dyn_mj_cfg = MuJoCoDynamicsParams(name="drone", f_plant=f_plant, f_true=f_true, nx=nx, nu=nu, u_lim=u_lim, mj_model=mj_model, mj_data=mj_data)

decentr_cfg_gauss = DecentralizedParams(N=N, Q=Q, R=R, H=H, term=False, mode="gauss-seidel")
decentr_cfg_jacbi = DecentralizedParams(N=N, Q=Q, R=R, H=H, term=False, mode="jacobi")
distr_cfg = DistributedParams(N=N, Q=Q, R=R, H=H, term=False)

rndzvs_env_cfg = RendezvousEnvParams(T=T, dt=dt, M=M, d_min=d_min, x0_val=x0_val, obs=obs)
ldr_env_cfg = LeaderEnvParams(T=T, dt=dt, M=M, d_min=d_min, x0_val=x0_val, xf_val_leader=xf_val[0, :], obs=obs)
env_cfg = EnvParams(T=T, dt=dt, M=M, d_min=d_min, x0_val=x0_val, xf_val=xf_val, obs=obs)

decentralized_rendezvous(dyn_mj_cfg, decentr_cfg_gauss, rndzvs_env_cfg, use_mj=True, vis_cfg=vis_cfg)
decentralized_rendezvous(dyn_mj_cfg, decentr_cfg_jacbi, rndzvs_env_cfg, use_mj=True, vis_cfg=vis_cfg)
distributed_rendezvous(dyn_mj_cfg, distr_cfg, rndzvs_env_cfg, use_mj=True, vis_cfg=vis_cfg)

decentralized_leader(dyn_mj_cfg, decentr_cfg_gauss, ldr_env_cfg, use_mj=True, vis_cfg=vis_cfg)
decentralized_leader(dyn_mj_cfg, decentr_cfg_jacbi, ldr_env_cfg, use_mj=True, vis_cfg=vis_cfg)
distributed_leader(dyn_mj_cfg, distr_cfg, ldr_env_cfg, use_mj=True, vis_cfg=vis_cfg)

decentralized(dyn_mj_cfg, decentr_cfg_gauss, env_cfg, use_mj=True, vis_cfg=vis_cfg)
decentralized(dyn_mj_cfg, decentr_cfg_jacbi, env_cfg, use_mj=True, vis_cfg=vis_cfg)
distributed(dyn_mj_cfg, distr_cfg, env_cfg, use_mj=True, vis_cfg=vis_cfg)