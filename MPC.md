# Model Predictive Control (MPC) Formulation for Visual Servoing

This document describes the mathematical principles and logic implemented in the `viper850-mpc-example.cpp` using the Aligator library.

## 1. General MPC Framework

Unlike a reactive QP controller, the MPC solves an Optimal Control Problem (OCP) over a finite time horizon $T$. It predicts the robot's future states and optimizes a sequence of control inputs $\mathbf{u} = \{\mathbf{u}_0, \dots, \mathbf{u}_{T-1}\}$ to minimize a cumulative cost while respecting constraints.

### Problem Definition
$$\min_{\mathbf{u}_{0 \dots T-1}} \sum_{t=0}^{T-1} \ell(\mathbf{x}_t, \mathbf{u}_t) + \ell_T(\mathbf{x}_T)$$
subject to:
- **Dynamics**: $\mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t \cdot \mathbf{u}_t$ (Linear discrete dynamics for joint positions $q$ and velocities $\dot{q}$)
- **Constraints**: $\mathbf{g}(\mathbf{x}_t, \mathbf{u}_t) \le 0$
- **Boxes**: $\mathbf{l}_{box} \le \mathbf{u}_t \le \mathbf{u}_{box}$

---

## 2. Cost Terms $\ell(\mathbf{x}, \mathbf{u})$

The MPC combines multiple objectives to ensure precise tracking and smooth motion.

### 2.1 Visual Servoing (Image Error)
The primary task is to minimize the distance between current image features $\mathbf{s}(q)$ and desired features $\mathbf{s}^*$:
$$\ell_{img} = \frac{1}{2} w_{img} \|\mathbf{s}(q) - \mathbf{s}^*\|^2$$
The weight $w_{img}$ is adapted online: it starts low to allow a smooth approach and increases as the error decreases to ensure high precision.

### 2.2 Pose Regularization
To avoid the inherent instabilities of IBVS (like the "camera retreat" phenomenon), a pose cost is added using the $SE(3)$ distance between the current and desired camera pose:
$$\ell_{pose} = \frac{1}{2} w_{pose} \|\log(C_{curr}^{-1} C_{des})\|^2$$

### 2.3 Control Effort (Regularization)
To prevent high-frequency oscillations and excessive joint velocities:
$$\ell_{ctrl} = \frac{1}{2} w_u \|\mathbf{u}\|^2$$

---

## 3. Constraints and Barrier Costs

The MPC handles critical limits (Singularities and Field of View) using a hybrid approach: **Hard Constraints** combined with **Soft Barrier Costs**.

### 3.1 Manipulability (Singularity Avoidance)
The goal is to maintain the manipulability index $m(q) = \sqrt{\det(J J^T)}$ above a minimum threshold $m_{min}$.

- **Hard Constraint**: $\mathbf{g}_{manip} = m_{min} - m(q) \le 0$. This guarantees the robot never enters a singularity.
- **Soft Barrier Cost**: A `RelaxedBarrierCost` is applied. When $m(q)$ drops below a threshold $m_{soft}$, the cost increases rapidly (logarithmically), pushing the robot back into the "safe" region.

### 3.2 Field of View (FoV)
Ensures the target points stay within the image boundaries $[u_{min}, u_{max}]$ and $[v_{min}, v_{max}]$ and remain in front of the camera ($Z \ge z_{min}$).

- **Hard Constraint**: $\mathbf{g}_{fov} \le 0$ (bounds on $u, v, Z$). This is a strict requirement to avoid losing track of the object.
- **Soft Barrier Cost**: A barrier cost is activated when points enter a "warning margin" (e.g., the outer 30% of the image). This prevents the robot from operating too close to the edges of the frame.

---

## 4. The Synergy of Constraints and Cost Terms

A key architectural choice in this MPC is the simultaneous use of both a hard constraint and a soft barrier cost for the same limit.

| Feature | Hard Constraint | Soft Barrier Cost |
| :--- | :--- | :--- |
| **Purpose** | Safety / Feasibility | Guidance / Optimization |
| **Effect** | Absolute boundary; cannot be crossed. | "Repulsive force" that steers the robot away. |
| **Numerical Impact** | Can cause solver failure if infeasible. | Smooths the optimization landscape. |
| **Behavior** | Corrects errors at the boundary. | Prevents errors from reaching the boundary. |

**Why use both?**
1. **Smoothness**: If only hard constraints were used, the robot would move freely until it hit the boundary, then suddenly change direction, leading to jerky motion. The barrier cost ensures a smooth transition.
2. **Robustness**: If only a cost term were used, a very strong primary task (image error) could potentially "push" the robot through the barrier into a singularity or out of the FoV. The hard constraint provides a final, unbreakable guarantee of safety.
3. **Convergence**: Barrier functions provide clear gradients that lead the solver toward the center of the feasible region, significantly improving the convergence rate of the ProxDPP solver.
