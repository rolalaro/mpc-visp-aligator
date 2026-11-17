// ibvs_mpc.cpp
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/stage-model.hpp>
#include <aligator/core/function-abstract.hpp>
#include <aligator/core/vector-space.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/costs/quad-residual-cost.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/linear-discrete-dynamics.hpp>
#include <aligator/modelling/linear-function.hpp>
#include <aligator/modelling/control-box-function.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/modelling/constraints/negative-orthant.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <chrono>
#include <iostream>
#include <memory>

namespace pin = pinocchio;
using namespace aligator;
using Scalar = double;
using VectorXs = Eigen::VectorXd;
using MatrixXs = Eigen::MatrixXd;
using StageFunction = StageFunctionTpl<Scalar>;
using StageFunctionData = StageFunctionDataTpl<Scalar>;
using ConstVectorRef = Eigen::Ref<const VectorXs>;

// ---------------------------------------------------------------------------
// Target shared by all residuals; update it every control cycle from vision.
// ---------------------------------------------------------------------------
struct VisualTarget
{
  Eigen::Vector3d p_w;   // 3D point in world frame (estimated from detection + depth)
  Eigen::Vector2d s_ref; // desired image point (normalized coords), (0,0) = centered
};

// ---------------------------------------------------------------------------
// Residual r(q) = s(q) - s*   (2 x nq Jacobian)
// ---------------------------------------------------------------------------
class ImagePointResidual : public StageFunction
{
public:
  using Base = StageFunction;
  using Data = StageFunctionData;

  ImagePointResidual(const pin::Model &model, pin::FrameIndex cam_frame,
                     std::shared_ptr<VisualTarget> target, int nu)
    : Base(model.nq, nu, 2), model_(model), pdata_(model), fid_(cam_frame),
    target_(std::move(target)), J_(6, model.nv)
  {
    J_.setZero();
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &,
                Data &data) const override
  {
    pin::forwardKinematics(model_, pdata_, x.head(model_.nq));
    pin::updateFramePlacement(model_, pdata_, fid_);
    const Eigen::Vector3d pc = pdata_.oMf[fid_].actInv(target_->p_w);
    const double Z = std::max(pc.z(), 1e-3); // avoid division by zero
    data.value_ << pc.x() / Z - target_->s_ref(0),
      pc.y() / Z - target_->s_ref(1);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &,
                        Data &data) const override
  {
    const Eigen::VectorXd q = x.head(model_.nq);
    pin::computeFrameJacobian(model_, pdata_, q, fid_, pin::LOCAL, J_);
    const Eigen::Vector3d pc = pdata_.oMf[fid_].actInv(target_->p_w);
    const double Z = std::max(pc.z(), 1e-3);

    // d p_c / dq = -J_v + [p_c]_x J_w   (camera-frame twist)
    const MatrixXs dpc = -J_.topRows<3>() +
      pin::skew(pc) * J_.bottomRows<3>();
    Eigen::Matrix<double, 2, 3> Ls;
    Ls << 1, 0, -pc.x() / Z,
      0, 1, -pc.y() / Z;
    Ls /= Z;

    data.Jx_ = Ls * dpc;   // nr x ndx
    data.Ju_.setZero();    // residual does not depend on u
  }

private:
  const pin::Model &model_;
  mutable pin::Data pdata_; // not thread-safe (fine for single-threaded solver)
  pin::FrameIndex fid_;
  std::shared_ptr<VisualTarget> target_;
  mutable MatrixXs J_;
};

// ---------------------------------------------------------------------------
// Control residual r(x,u) = u  (used to build the cost 1/2 u^T R u)
// ---------------------------------------------------------------------------
class ControlResidual : public StageFunction
{
public:
  using Base = StageFunction;
  using Data = StageFunctionData;

  ControlResidual(int ndx, int nu) : Base(ndx, nu, nu) {}

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                Data &data) const override
  {
    data.value_ = u;
  }

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        Data &data) const override
  {
    data.Jx_.setZero();
    data.Ju_.setIdentity();
  }
};

// ---------------------------------------------------------------------------
// Constraint g(q) = w_min - w(q) <= 0, with w = sqrt(det(J J^T)) (Yoshikawa)
// ---------------------------------------------------------------------------
class ManipulabilityConstraint : public StageFunction
{
public:
  using Base = StageFunction;
  using Data = StageFunctionData;

  ManipulabilityConstraint(const pin::Model &model, pin::FrameIndex ee_frame,
                           double w_min, int nu)
    : Base(model.nq, nu, 1), model_(model), pdata_(model), fid_(ee_frame),
    w_min_(w_min), J_(6, model.nv)
  {
    J_.setZero();
  }

  double manipulability(const VectorXs &q) const
  {
    pin::computeFrameJacobian(model_, pdata_, q, fid_, pin::LOCAL, J_);
    const double det = (J_ * J_.transpose()).determinant();
    return std::sqrt(std::max(det, 0.0));
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &,
                Data &data) const override
  {
    data.value_[0] = w_min_ - manipulability(x.head(model_.nq));
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &,
                        Data &data) const override
  {
    const double eps = 1e-6;
    VectorXs q = x.head(model_.nq);
    for (int i = 0; i < model_.nv; ++i) {
      VectorXs qp = q, qm = q;
      qp[i] += eps;
      qm[i] -= eps;
      data.Jx_(0, i) = -(manipulability(qp) - manipulability(qm)) / (2 * eps);
    }
    data.Ju_.setZero();
  }

private:
  const pin::Model &model_;
  mutable pin::Data pdata_;
  pin::FrameIndex fid_;
  double w_min_;
  mutable MatrixXs J_;
};

// ---------------------------------------------------------------------------
// Field-of-view constraint g(q) <= 0 on the target point in the camera frame
//   rows: X - xmax Z, -X - xmax Z, Y - ymax Z, -Y - ymax Z, zmin - Z
// ---------------------------------------------------------------------------
class FieldOfViewConstraint : public StageFunction
{
public:
  using Base = StageFunction;
  using Data = StageFunctionData;

  FieldOfViewConstraint(const pin::Model &model, pin::FrameIndex cam_frame,
                        std::shared_ptr<VisualTarget> target, double x_max,
                        double y_max, double z_min, int nu)
    : Base(model.nq, nu, 5), model_(model), pdata_(model), fid_(cam_frame),
    target_(std::move(target)), J_(6, model.nv)
  {
    J_.setZero();
    C_ << 1, 0, -x_max,
      -1, 0, -x_max,
      0, 1, -y_max,
      0, -1, -y_max,
      0, 0, -1;
    d_ << 0, 0, 0, 0, z_min;
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &,
                Data &data) const override
  {
    pin::forwardKinematics(model_, pdata_, x.head(model_.nq));
    pin::updateFramePlacement(model_, pdata_, fid_);
    const Eigen::Vector3d pc = pdata_.oMf[fid_].actInv(target_->p_w);
    data.value_ = C_ * pc + d_;
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &,
                        Data &data) const override
  {
    const Eigen::VectorXd q = x.head(model_.nq);
    pin::computeFrameJacobian(model_, pdata_, q, fid_, pin::LOCAL, J_);
    const Eigen::Vector3d pc = pdata_.oMf[fid_].actInv(target_->p_w);

    // d p_c / dq = -J_v + [p_c]_x J_w
    const MatrixXs dpc = -J_.topRows<3>() + pin::skew(pc) * J_.bottomRows<3>();

    data.Jx_ = C_ * dpc; // 5 x nv
    data.Ju_.setZero();
  }

private:
  const pin::Model &model_;
  mutable pin::Data pdata_;
  pin::FrameIndex fid_;
  std::shared_ptr<VisualTarget> target_;
  mutable MatrixXs J_;
  Eigen::Matrix<double, 5, 3> C_;
  Eigen::Matrix<double, 5, 1> d_;
};

// ---------------------------------------------------------------------------
// Relaxed-log-barrier cost on any constraint function g(x,u) <= 0.
//   h_i = -g_i / scale   (h_i > 0  <=>  constraint satisfied)
//   B(h) = -mu log h                                   if h > delta
//        = mu/2 [((h-2delta)/delta)^2 - 1] - mu log delta   otherwise
// ---------------------------------------------------------------------------
template <class Func>
class RelaxedBarrierCost : public CostAbstractTpl<Scalar>
{
public:
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  RelaxedBarrierCost(const xyz::polymorphic<Manifold> &space, const Func &f,
                     double mu, double delta, double scale = 1.0)
    : Base(space, f.nu), f_(f), fdata_(f_.createData()), mu_(mu),
    delta_(delta), inv_scale_(1.0 / scale)
  {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data) const override
  {
    refresh(x, u, false);
    double v = 0.0;
    for (int i = 0; i < fdata_->value_.size(); ++i)
      v += barrier(h(i));
    data.value_ = v;
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data) const override
  {
    refresh(x, u, true);
    data.Lx_.setZero();
    data.Lu_.setZero();
    for (int i = 0; i < fdata_->value_.size(); ++i)
      // dh/dx = -inv_scale * dg/dx
      data.Lx_.noalias() +=
      (-dBarrier(h(i)) * inv_scale_) * fdata_->Jx_.row(i).transpose();
  }

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       CostData &data) const override
  {
    refresh(x, u, true);
    data.Lxx_.setZero();
    data.Lxu_.setZero();
    data.Luu_.setZero();
    const double s2 = inv_scale_ * inv_scale_;
    for (int i = 0; i < fdata_->value_.size(); ++i) {
      const auto gi = fdata_->Jx_.row(i);
      data.Lxx_.noalias() += (d2Barrier(h(i)) * s2) * gi.transpose() * gi;
    }
  }

private:
  double h(int i) const { return -fdata_->value_[i] * inv_scale_; }

  // Evaluate (and optionally differentiate) g, skipping work if (x,u) is
  // unchanged. Matters for the finite-difference manipulability Jacobian.
  void refresh(const ConstVectorRef &x, const ConstVectorRef &u,
               bool need_jac) const
  {
    const bool same = valid_ && x_last_.size() == x.size() &&
      x_last_ == x && u_last_ == u;
    if (same && (jac_ok_ || !need_jac)) return;
    f_.evaluate(x, u, *fdata_);
    if (need_jac) f_.computeJacobians(x, u, *fdata_);
    x_last_ = x; u_last_ = u;
    valid_ = true;
    jac_ok_ = need_jac;
  }

  double barrier(double h) const
  {
    if (h > delta_) return -mu_ * std::log(h);
    const double r = (h - 2.0 * delta_) / delta_;
    return 0.5 * mu_ * (r * r - 1.0) - mu_ * std::log(delta_);
  }
  double dBarrier(double h) const
  {
    if (h > delta_) return -mu_ / h;
    return mu_ * (h - 2.0 * delta_) / (delta_ * delta_);
  }
  double d2Barrier(double h) const
  {
    if (h > delta_) return mu_ / (h * h);
    return mu_ / (delta_ * delta_);
  }

  Func f_;
  mutable decltype(f_.createData()) fdata_;
  double mu_, delta_, inv_scale_;
  mutable VectorXs x_last_, u_last_;
  mutable bool valid_ = false, jac_ok_ = false;
};

using FovBarrierCost = RelaxedBarrierCost<FieldOfViewConstraint>;
using ManipBarrierCost = RelaxedBarrierCost<ManipulabilityConstraint>;

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  if (argc < 3) {
    std::cerr << "usage: ibvs_mpc robot.urdf camera_frame_name\n";
    return 1;
  }

  // ---- Robot model --------------------------------------------------------
  pin::Model model;
  pin::urdf::buildModel(argv[1], model);
  const pin::FrameIndex cam_fid = model.getFrameId(argv[2]);
  const int nq = model.nq, nu = model.nv;
  const double dt = 0.02;   // 50 Hz MPC
  const int horizon = 25;   // 0.5 s lookahead

  // ---- Parameters ---------------------------------------------------------
  const double w_min = 0.02;  // minimum manipulability (tune! must be < initial w)
  const double w_soft = 2.5 * w_min;    // barrier starts to act around here
  const double w_img = 1e3;   // image error weight
  const double w_img_term = 1e4;
  const double w_u = 1e-2;    // velocity regularization
  const double barrier_mu = 5e-2;     // barrier strength
  const double barrier_delta = 0.03;  // relaxation threshold (units of g: meters)
  const double manip_mu = 5e-2;
  const double manip_delta = 0.2;       // normalized: relax when w < 1.2 * w_soft

  // Camera intrinsics (replace with your calibration)
  const double fx = 600.0, fy = 600.0;
  const double img_w = 640.0, img_h = 480.0;
  const double fov_margin = 0.8;  // keep the point within 80% of the half-image
  const double x_max = fov_margin * (img_w / 2.0) / fx;
  const double y_max = fov_margin * (img_h / 2.0) / fy;
  const double z_min = 0.15;      // minimum distance in front of the camera [m]

  // ---- Initial state and visual target ------------------------------------
  VectorXs q0 = pin::neutral(model);          // replace with your actual configuration
  q0 = 0.5 * (model.lowerPositionLimit + model.upperPositionLimit);

  pin::Data tmp(model);
  pin::forwardKinematics(model, tmp, q0);
  pin::updateFramePlacements(model, tmp);
  auto target = std::make_shared<VisualTarget>();
  // Demo: a point 0.6 m in front of the camera, off-center
  target->p_w = tmp.oMf[cam_fid].act(Eigen::Vector3d(0.1, -0.05, 0.6));
  target->s_ref.setZero();

  // ---- Spaces, dynamics ---------------------------------------------------
  auto space = VectorSpaceTpl<Scalar>(nq);

  MatrixXs A = MatrixXs::Identity(nq, nq);
  MatrixXs B = dt * MatrixXs::Identity(nq, nu);
  VectorXs c = VectorXs::Zero(nq);
  dynamics::LinearDiscreteDynamicsTpl<Scalar> dyn(A, B, c); // q+ = q + dt*u

    // ---- Constraints ----------------------------------------------------------
    // q limits:  [I; -I] q + [-qmax; qmin] <= 0
  MatrixXs Aq(2 * nq, nq);
  Aq << MatrixXs::Identity(nq, nq), -MatrixXs::Identity(nq, nq);
  VectorXs cq(2 * nq);
  cq << -model.upperPositionLimit, model.lowerPositionLimit;
  LinearFunctionTpl<Scalar> q_limits(Aq, MatrixXs::Zero(2 * nq, nu), cq);

  // velocity box
  VectorXs umax = model.velocityLimit;
  ControlBoxFunctionTpl<Scalar> u_box(nq, -umax, umax);


  ManipulabilityConstraint manip(model, cam_fid, w_min, nu);        // hard
  ManipulabilityConstraint manip_soft(model, cam_fid, w_soft, nu);  // for barrier
  FieldOfViewConstraint fov(model, cam_fid, target, x_max, y_max, z_min, nu);
  const double barrier_margin = 0.7;
  FieldOfViewConstraint fov_soft(model, cam_fid, target,
      barrier_margin * (img_w / 2.0) / fx,
      barrier_margin * (img_h / 2.0) / fy, z_min, nu);

  // ---- Cost ----------------------------------------------------------------
  ImagePointResidual img_res(model, cam_fid, target, nu);
  ControlResidual ctrl_res(nq, nu);
  const MatrixXs R = w_u * MatrixXs::Identity(nu, nu);

  auto make_cost = [&](double wimg, bool with_u) {
    CostStackTpl<Scalar> cs(space, nu);
    cs.addCost(QuadraticResidualCostTpl<Scalar>(
      space, img_res, wimg * Eigen::Matrix2d::Identity()));
    if (with_u)
      cs.addCost(QuadraticResidualCostTpl<Scalar>(space, ctrl_res, R)); // <-- replaces QuadraticControlCost
    cs.addCost(FovBarrierCost(space, fov_soft, barrier_mu, barrier_delta));
    cs.addCost(ManipBarrierCost(space, manip_soft, manip_mu, manip_delta,
                                /*scale=*/w_soft));
    return cs;
    };

  // ---- Build the OCP --------------------------------------------------------
  using aligator::NegativeOrthantTpl;
  std::vector<xyz::polymorphic<StageModelTpl<Scalar>>> dummy; // (not needed)
  TrajOptProblemTpl<Scalar> problem(q0, nu, space, make_cost(w_img_term, false));

  for (int t = 0; t < horizon; ++t) {
    StageModelTpl<Scalar> stage(make_cost(w_img, true), dyn);
    stage.addConstraint(q_limits, NegativeOrthantTpl<Scalar>());
    stage.addConstraint(u_box, NegativeOrthantTpl<Scalar>());
    stage.addConstraint(manip, NegativeOrthantTpl<Scalar>());
    stage.addConstraint(fov, NegativeOrthantTpl<Scalar>());
    problem.addStage(stage);
  }
  // Also constrain the terminal state
  problem.addTerminalConstraint(q_limits, NegativeOrthantTpl<Scalar>());
  problem.addTerminalConstraint(manip, NegativeOrthantTpl<Scalar>());
  problem.addTerminalConstraint(fov, NegativeOrthantTpl<Scalar>());

  // ---- Solver ----------------------------------------------------------------
  const double tol = 1e-4, mu_init = 1e-2;
  const std::size_t max_iters = 20;
  SolverProxDDPTpl<Scalar> solver(tol, mu_init, max_iters, QUIET);
  solver.setup(problem);

  std::vector<VectorXs> xs(horizon + 1, q0);
  std::vector<VectorXs> us(horizon, VectorXs::Zero(nu));

  // ---- MPC loop (simulation) ---------------------------------------------------
  VectorXs q = q0;
  for (int k = 0; k < 300; ++k) {
    // 1) update the problem with current measurement
    problem.setInitState(q);          // older versions: problem.x0_init_ = q;
    // target->p_w = ...  (update from vision here)

    // 2) solve, warm-started
    xs[0] = q;
    auto t0 = std::chrono::steady_clock::now();
    solver.run(problem, xs, us);
    auto t1 = std::chrono::steady_clock::now();
    const auto &res = solver.results_;
    const VectorXs u0 = res.us[0];

    // 3) apply first control (here: integrate the simulated robot)
    q += dt * u0;

    // 4) shift the solution for the next warm start
    for (int t = 0; t < horizon - 1; ++t) us[t] = res.us[t + 1];
    us[horizon - 1] = res.us[horizon - 1];
    for (int t = 0; t < horizon; ++t) xs[t] = res.xs[t + 1];
    xs[horizon] = res.xs[horizon];

    // monitoring
    pin::forwardKinematics(model, tmp, q);
    pin::updateFramePlacement(model, tmp, cam_fid);
    const Eigen::Vector3d pc = tmp.oMf[cam_fid].actInv(target->p_w);
    const double Z = pc.z();
    const double fov_slack = std::min({ x_max - pc.x() / Z, x_max + pc.x() / Z,
      y_max - pc.y() / Z, y_max + pc.y() / Z });
    std::cout << "k=" << k << " solve_ms=" << std::chrono::duration<double, std::milli>(t1 - t0).count() << " iters=" << res.num_iters
      << "  s=(" << pc.x() / pc.z() << ", "
      << pc.y() / pc.z() << ")" << "  fov_slack=" << fov_slack << "  w="
      << manip.manipulability(q) << "  conv=" << res.conv << "\n";
  }
}
