# Quadratic Programming (QP) Formulation for Visual Servoing

This document describes the mathematical formulation of the Quadratic Programming (QP) problem used in the `viper850-qp-example` to perform Image-Based Visual Servoing (IBVS) while avoiding singularities, maintaining the Field of View (FoV), and respecting robot joint limits.

## 1. General QP Problem

The control law is formulated as a QP problem solved at each time step to find the optimal joint velocities $\dot{q}$. To allow for "soft" constraints (constraints that can be slightly violated if necessary to maintain stability or feasibility), we introduce slack variables $s$.

The state vector of the QP is defined as:
$$\mathbf{x} = \begin{bmatrix} \dot{q} \\ s_{manip} \\ s_{fov} \end{bmatrix} \in \mathbb{R}^{nu + 2}$$
where $nu$ is the number of robot joints.

The problem is:
$$\min_{\mathbf{x}} \frac{1}{2} \mathbf{x}^T \mathbf{H} \mathbf{x} + \mathbf{g}^T \mathbf{x}$$
subject to:
$$\mathbf{C}\mathbf{x} \le \mathbf{u}$$
$$\mathbf{l}_{box} \le \mathbf{x} \le \mathbf{u}_{box}$$

---

## 2. Cost Function $\mathbf{H}$ and $\mathbf{g}$

The cost function consists of three main components:

### 2.1 Visual Servoing (Task)
The objective is to minimize the error $\mathbf{e} = \mathbf{s} - \mathbf{s}^*$ between current and desired image features. Using a damped least-squares approach:
$$\text{Cost}_{task} = \frac{1}{2} \dot{q}^T (J_{task}^T J_{task} + \alpha I) \dot{q} - (\lambda J_{task}^T \mathbf{e})^T \dot{q}$$
where $J_{task} = L_s cVe eJe$ is the interaction matrix mapping joint velocities to feature velocities.

### 2.2 Manipulability (Singularity Avoidance)
To avoid singularities, we maximize the manipulability index $m(q) = \sqrt{\det(J J^T)}$. We add a penalty for the slack variable and a gradient descent term to pull the robot toward higher manipulability:
$$\text{Cost}_{manip} = \frac{1}{2} w_{manip} s_{manip}^2 - (\beta w_{manip} \nabla m(q))^T \dot{q}$$

### 2.3 Field of View (FoV)
To ensure that feature points stay within the image, we penalize the FoV slack variable:
$$\text{Cost}_{fov} = \frac{1}{2} w_{fov} s_{fov}^2$$

---

## 3. Constraints

### 3.1 Manipulability Soft Constraint
We want to ensure $m(q) \ge m_{thresh}$. Linearizing the manipulability index:
$$m(q_{t+1}) \approx m(q_t) + \nabla m(q)^T \dot{q} \Delta t \ge m_{thresh}$$
Rearranging into the form $\mathbf{C}\mathbf{x} \le \mathbf{u}$ and adding slack $s_{manip}$:
$$-\nabla m(q)^T \dot{q} - s_{manip} \le \frac{m(q_t) - m_{thresh}}{\Delta t}$$
Additionally, we ensure the slack is non-negative:
$$-s_{manip} \le 0$$

### 3.2 Field of View (FoV) Soft Constraints
For each feature point $(u_i, v_i)$, we want it to remain within the margins $[u_{min}, u_{max}]$ and $[v_{min}, v_{max}]$.
The motion of the points is $\dot{s} = J_{task} \dot{q}$. For each point $i \in \{0, \dots, 3\}$:

**Upper bounds:**
$$J_{u,i} \dot{q} - s_{fov} \le \frac{u_{max} - u_i}{\Delta t}$$
$$J_{v,i} \dot{q} - s_{fov} \le \frac{v_{max} - v_i}{\Delta t}$$

**Lower bounds:**
$$-J_{u,i} \dot{q} - s_{fov} \le \frac{u_i - u_{min}}{\Delta t}$$
$$-J_{v,i} \dot{q} - s_{fov} \le \frac{v_i - v_{min}}{\Delta t}$$

And the slack non-negativity:
$$-s_{fov} \le 0$$

### 3.3 Joint Limits (Hard Constraints)
Joint velocity limits $\dot{q} \in [-\dot{q}_{max}, \dot{q}_{max}]$ and position limits $q \in [q_{min}, q_{max}]$ are handled using the **Velocity Damper** method. The joint velocity limits are tightened as the robot approaches a position limit:
$$\dot{q}_{upper} = \min \left( \dot{q}_{max}, \frac{q_{max} - q}{\eta} \right)$$
$$\dot{q}_{lower} = \max \left( -\dot{q}_{max}, \frac{q_{min} - q}{\eta} \right)$$
These are implemented as box constraints: $\mathbf{l}_{box} \le \dot{q} \le \mathbf{u}_{box}$.
