# Distributed & Decentralized MPC

This project implements Distributed and Decentralized Model Predictive Control (DMPC) strategies for multi-agent systems using **CasADi**.

<img src="media/thumbnail.gif" width="475">

## Features

### Control Architectures
There are two main architectures for DMPC implemented
- **Distributed**: Single, centralized solver solves a global optimization problem. More computationally intensive but may yield better performance.
- **Decentralized**: Agents solve local optimization problems and share solutions. Less computationally intensive but may yield suboptimal performance.
  - *Jacobi*: Agents' solutions are shared after each horizon iteration.
  - *Gauss-Seidel*: Agents' solutions are shared immediately after each agent's update.
  
Both architectures support:
- **Standard**: Each agent plans its trajectory independently.
- **Leader-Follower**: One agent (leader) plans its trajectory, and followers plan their trajectories to track the leader.
- **Rendezvous**: Agents rendezvous with one another.

### Dynamics
#### Models
- *Drone*: 12-state quadrotor dynamics.
- *Bicycle*: 5-state kinematic bicycle model.
- *Custom*: Specify your own dynamics.

#### Integrators
- *Forward Euler*
- *Runge-Kutta (RK4)*
- *Custom*: Specify your own integrator.

#### Plant
- `f` is a CasADi function for the **plant model** used by the MPC solver. 
- `f_np` is a NumPy function for the **actual plant dynamics**.
- Note: `f` and `f_np` may differ in terms of integrator choice, model fidelity (e.g., nonlinear vs. linear, full-order vs reduced-order), stochasticity, etc. 
  - In general, `f_np` should implement a high-fidelity representation of the system dynamics that captures real-world dynamics, including stochasticity, disturbances, uncertainties, etc. This is the model representing the true system.
  - By contrast, `f` can implement a low-fidelity, high-fidelity, reduced-order, or virtual representation of the system dynamics that is computationally efficient for real-time control. This is the model that the MPC solver interacts with.

### Constraints
- **Control constraints**: Specify minimum and maximum control inputs.
- **Inter-agent collision avoidance**: Specify minimum distance between agents.
- **Obstacle avoidance**: Customizable static obstacles in the environment. Dynamic obstacles planned.

### State Estimation
Planned.

## Installation

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate dmpc
```

## Usage

Run the `<model>.py` scripts. Plots are saved to the `plots/` directory.
```bash
python3 -m scripts.bicycle # Bicycle dynamics
python3 -m scripts.drone   # Drone dynamics
python3 -m scripts.double_integrator # Double integrator dynamics
```

## Structure

`scripts/`: Example scripts for different dynamics models.
- `bicycle.py`: Example script for bicycle dynamics.
- `drone.py`: Example script for drone dynamics.
- `double_integrator.py`: Example script for double integrator dynamics.

`src/`: Source code for DMPC implementations.
- `distributed_mpc.py`: Distributed MPC implementation.
- `decentralized_mpc.py`: Decentralized MPC implementation (Jacobi and Gauss-Seidel).
- `distributed_mpc_leader.py`: Distributed MPC with leader-follower architecture.
- `decentralized_mpc_leader.py`: Decentralized MPC with leader-follower architecture.
- `distributed_mpc_rendezvous.py`: Distributed MPC for rendezvous tasks.
- `decentralized_mpc_rendezvous.py`: Decentralized MPC for rendezvous tasks.

`utils/`: Plotting utilities.

`run.sh`: Shell script to re-run example scripts repeatedly until all simulations complete successfully.
