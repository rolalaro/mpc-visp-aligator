# Implementing Hybrid Visual Servoing in MPC

To reduce the pose error (`cdMc`) while maintaining the convergence of the image points, you can add a **Pose Cost** to your optimization problem. This transforms the system into a Hybrid Visual Servoing controller.

## 1. Define the Pose Residual
Add this class to your code. It computes the error between the current camera pose and the desired pose using the $\text{se}(3)$ logarithm (the "twist" that takes you from current to desired).

```cpp
class PoseResidual : public StageFunction
{
public:
  using Base = StageFunction;
  using Data = StageFunctionData;

  PoseResidual(const pin::Model &model, pin::FrameIndex cam_frame,
               const pin::SE3 &target_pose, int nu)
    : Base(model.nq, nu, 6), model_(model), pdata_(model), fid_(cam_frame),
      target_pose_(target_pose), J_(6, model.nv)
  {
    J_.setZero();
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &,
                Data &data) const override
  {
    pin::forwardKinematics(model_, pdata_, x.head(model_.nq));
    pin::updateFramePlacement(model_, pdata_, fid_);
    const auto &fMc = pdata_.oMf[fid_];

    // Compute the relative pose: C_rel = C_curr^-1 * C_des
    // This is the pose of the target relative to the current camera.
    pin::SE3 rel = fMc.inverse() * target_pose_;
    
    // Use the log map to get the 6D twist in the local frame
    data.value_ = pin::log6(rel).vector();
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &,
                        Data &data) const override
  {
    const Eigen::VectorXd q = x.head(model_.nq);
    pin::computeJointJacobians(model_, pdata_, q);
    pin::updateFramePlacement(model_, pdata_, fid_);
    
    // For the residual log(C_curr^-1 * C_des), the Jacobian is exactly
    // the local frame Jacobian.
    pin::getFrameJacobian(model_, pdata_, fid_, pin::LOCAL, J_);
    
    data.Jx_ = J_;
    data.Ju_.setZero();
  }

private:
  const pin::Model &model_;
  mutable pin::Data pdata_;
  pin::FrameIndex fid_;
  pin::SE3 target_pose_;
  mutable MatrixXs J_;
};
```

## 2. Integrate into the MPC setup

### Update Parameters
Add a weight for the pose cost.
```cpp
double w_pose = 1e1; // Start small, then increase to pull the pose closer
```

### Update `make_cost`
Include the `PoseResidual` in the cost stack.
```cpp
// Create the residual once outside the loop
PoseResidual pose_res(model, cam_fid, pin::SE3(cdMo.matrix()), nu);

auto make_cost = [&](bool with_u) {
  CostStackTpl<Scalar> cs(space, nu);
  
  // 1. Image Cost (already implemented)
  cs.addCost(WeightedResidualCost<ImagePointResidual>(space, img_res, w_img));
  
  // 2. Pose Cost (New!)
  cs.addCost(WeightedResidualCost<PoseResidual>(space, pose_res, w_pose));
  
  if (with_u)
    cs.addCost(WeightedResidualCost<ControlResidual>(space, ctrl_res, w_u));
    
  // ... (barriers)
  return cs;
};
```

## 3. Tuning Tips
Since you now have two competing costs (Image vs Pose), you need to balance them:

1.  **Image-Dominant**: If `w_img` $\gg$ `w_pose`, the robot will prioritize the pixels. This is your current state.
2.  **Pose-Dominant**: If `w_pose` $\gg$ `w_img`, the robot will try to reach the 3D pose regardless of the image. This can lead to "blind" movements if the camera parameters are slightly off.
3.  **Balanced**: Start with a very small `w_pose` and increase it gradually until the `cdMc` error decreases without disturbing the 0.5-pixel image convergence.

**Note**: If you notice the "jitter" increases, you can also apply the adaptive logic to `w_pose` (e.g., increasing it as the image error decreases).
