/*
 * ViSP, open source Visual Servoing Platform software.
 * Copyright (C) 2005 - 2025 by Inria. All rights reserved.
 *
 * This software is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * See the file LICENSE.txt at the root directory of this source
 * distribution for additional information about the GNU GPL.
 *
 * For using ViSP with software that can not be combined with the GNU
 * GPL, please contact Inria about acquiring a ViSP Professional
 * Edition License.
 *
 * See https://visp.inria.fr for more information.
 *
 * This software was developed at:
 * Inria Rennes - Bretagne Atlantique
 * Campus Universitaire de Beaulieu
 * 35042 Rennes Cedex
 * France
 *
 * If you have questions regarding the use of this file, please contact
 * Inria at visp@inria.fr
 *
 * This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
 * WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Description:
 * Simulation of a 2D visual servoing using 4 points with polar
 * coordinates as visual feature.
 */

/*!
  \example servoSimuViper850FourPoints2DCamVelocity.cpp

  \brief Simulation of a 2D visual servoing:

  Simulation of a 2D visual servoing:
  - servo on 4 points with cartesian coordinates,
  - eye-in-hand control law,
  - camera velocities are computed,
  - display internal camera view and an external view.

  Interaction matrix is computed as the mean of the current and desired
  interaction matrix.

*/

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpDebug.h>

#if defined(VISP_HAVE_THREADS) && defined(VISP_HAVE_DISPLAY) \
  && (defined(VISP_HAVE_LAPACK) || defined(VISP_HAVE_EIGEN3) || defined(VISP_HAVE_OPENCV))

// We need to use threading capabilities. Thus on Unix-like
// platforms, the libpthread third-party library need to be
// installed. On Windows, we use the native threading capabilities.

#include <ctime>
#include <stdio.h>
#include <stdlib.h>

#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpEigenConversion.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpImage.h>
#include <visp3/core/vpImagePoint.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpMath.h>
#include <visp3/core/vpMeterPixelConversion.h>
#include <visp3/gui/vpDisplayFactory.h>
#include <visp3/io/vpParseArgv.h>
#include <visp3/robot/vpSimulatorViper850.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>

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
#include <aligator/solvers/fddp/solver-fddp.hpp>
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
  std::vector<Eigen::Vector3d> p_f;   // N points, world frame
  std::vector<Eigen::Vector2d> s_ref; // N desired normalized image points
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
    : Base(model.nq, nu, 2 * static_cast<int>(target->p_f.size())), model_(model), pdata_(model), fid_(cam_frame),
    target_(std::move(target)), J_(6, model.nv)
  {
    J_.setZero();
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &,
                Data &data) const override
  {
    pin::forwardKinematics(model_, pdata_, x.head(model_.nq));
    pin::updateFramePlacement(model_, pdata_, fid_);
    const auto &fMc = pdata_.oMf[fid_];
    for (std::size_t i = 0; i < target_->p_f.size(); ++i) {
      const Eigen::Vector3d pc = fMc.actInv(target_->p_f[i]);
      const double Z = std::max(pc.z(), 1e-3);
      data.value_.segment<2>(2 * i) =
        Eigen::Vector2d(pc.x() / Z, pc.y() / Z) - target_->s_ref[i];
    }
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &,
                        Data &data) const override
  {
    const Eigen::VectorXd q = x.head(model_.nq);
    pin::computeJointJacobians(model_, pdata_, q);   // runs FK for this q
    pin::updateFramePlacement(model_, pdata_, fid_); // refresh oMf for this q
    pin::getFrameJacobian(model_, pdata_, fid_, pin::LOCAL, J_);
    const auto &fMc = pdata_.oMf[fid_];              // now consistent with J_
    data.Jx_.setZero();
    data.Ju_.setZero();
    for (std::size_t i = 0; i < target_->p_f.size(); ++i) {
      const Eigen::Vector3d pc = fMc.actInv(target_->p_f[i]);
      const double Z = std::max(pc.z(), 1e-3);
      Eigen::Matrix<double, 2, 3> Ls;
      Ls << 1, 0, -pc.x() / Z,
        0, 1, -pc.y() / Z;
      Ls /= Z;
      // d p_c / dq = -J_v + [p_c]_x J_w
      data.Jx_.middleRows<2>(2 * i).noalias() =
        Ls * (-J_.topRows<3>() + pin::skew(pc) * J_.bottomRows<3>());
    }
    data.Ju_.setZero();    // residual does not depend on u
  }

private:
  const pin::Model &model_;
  mutable pin::Data pdata_; // not thread-safe (fine for single-threaded solver)
  pin::FrameIndex fid_;
  std::shared_ptr<VisualTarget> target_;
  mutable MatrixXs J_;
};

class PoseResidual : public StageFunction
{
public:
  using Base = StageFunction;
  using Data = StageFunctionData;

  PoseResidual(const pin::Model &model, pin::FrameIndex cam_frame,
               const pin::SE3 &target_pose, int nu)
    : Base(model.nq, nu, 6), model_(model), pdata_(model), fid_(cam_frame),
    cd_M_f_(target_pose), J_(6, model.nv)
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
    pin::SE3 err = cd_M_f_ * fMc;

    // Use the log map to get the 6D twist (translation and rotation vector)
    data.value_ = pin::log6(err).toVector();
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
  pin::SE3 cd_M_f_;
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
    : Base(model.nq, nu, 5 * static_cast<int>(target->p_f.size())), model_(model), pdata_(model), fid_(cam_frame),
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
    const auto &fMc = pdata_.oMf[fid_];
    for (std::size_t i = 0; i < target_->p_f.size(); ++i) {
      const Eigen::Vector3d pc = fMc.actInv(target_->p_f[i]);
      data.value_.segment<5>(5 * i) = C_ * pc + d_;
    }
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &,
                        Data &data) const override
  {
    data.Jx_.setZero();
    data.Ju_.setZero();

    const Eigen::VectorXd q = x.head(model_.nq);
    pin::computeJointJacobians(model_, pdata_, q);   // runs FK for this q
    pin::updateFramePlacement(model_, pdata_, fid_); // refresh oMf for this q
    pin::getFrameJacobian(model_, pdata_, fid_, pin::LOCAL, J_);
    const auto &fMc = pdata_.oMf[fid_];              // now consistent with J_
    for (std::size_t i = 0; i < target_->p_f.size(); ++i) {
      const Eigen::Vector3d pc = fMc.actInv(target_->p_f[i]);
      // d p_c / dq = -J_v + [p_c]_x J_w
      const MatrixXs dpc =
        -J_.topRows<3>() + pin::skew(pc) * J_.bottomRows<3>();
      data.Jx_.middleRows<5>(5 * i).noalias() = C_ * dpc;
    }
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
// Cost function: 1/2 * w * ||r(x,u)||^2
// ---------------------------------------------------------------------------
template <class Res>
class WeightedResidualCost : public CostAbstractTpl<Scalar>
{
public:
  using Base = CostAbstractTpl<Scalar>;

  WeightedResidualCost(const xyz::polymorphic<Manifold> &space, Res &res, double &weight)
    : Base(space, res.nu), res_(res), weight_(weight)
  {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data) const override
  {
    auto d = res_.createData();
    res_.evaluate(x, u, *d);
    data.value_ = 0.5 * weight_ * d->value_.squaredNorm();
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data) const override
  {
    auto d = res_.createData();
    res_.evaluate(x, u, *d);
    res_.computeJacobians(x, u, *d);
    data.Lx_.noalias() = weight_ * d->Jx_.transpose() * d->value_;
    data.Lu_.noalias() = weight_ * d->Ju_.transpose() * d->value_;
  }

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data) const override
  {
    auto d = res_.createData();
    res_.evaluate(x, u, *d);
    res_.computeJacobians(x, u, *d);
    data.Lxx_.noalias() = weight_ * d->Jx_.transpose() * d->Jx_;
    data.Lxu_.noalias() = weight_ * d->Jx_.transpose() * d->Ju_;
    data.Luu_.noalias() = weight_ * d->Ju_.transpose() * d->Ju_;
    // Note: we ignore the second-order term (Gauß-Newton approximation)
  }

private:
  Res &res_;
  double &weight_;
};

// ---------------------------------------------------------------------------
// Relaxed-log-barrier cost on any constraint function g(x,u) <= 0.
//   h_i = -g_i / scale   (h_i > 0  <=>  constraint satisfied)
//   B(h) = μ ( h/h_on − 1 − log(h/h_on) )   for delta < h < h_on     (B = B' = 0 at h_on)
//   B(h) = 0                                 for h ≥ h_on
//   B(h) = quadratic extension of the above  for h ≤ delta           (matches value, slope, curvature at delta)
// ---------------------------------------------------------------------------
template <class Func>
class RelaxedBarrierCost : public CostAbstractTpl<Scalar>
{
public:
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  RelaxedBarrierCost(const xyz::polymorphic<Manifold> &space, const Func &f,
                     double mu, double delta, double h_on, double scale = 1.0)
    : Base(space, f.nu), f_(f), fdata_(f_.createData()), mu_(mu),
    delta_(delta), inv_scale_(1.0 / scale), h_on_(h_on)
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
    // if (same && (jac_ok_ || !need_jac)) return;
    f_.evaluate(x, u, *fdata_);
    if (need_jac) f_.computeJacobians(x, u, *fdata_);
    x_last_ = x; u_last_ = u;
    valid_ = true;
    jac_ok_ = need_jac;
  }

  double barrier(double h) const
  {
    if (h >= h_on_) return 0.0;
    if (h > delta_) return mu_ * (h / h_on_ - 1.0 - std::log(h / h_on_));
    const double Bd = mu_ * (delta_ / h_on_ - 1.0 - std::log(delta_ / h_on_));
    const double dBd = mu_ * (1.0 / h_on_ - 1.0 / delta_);
    const double d2 = mu_ / (delta_ * delta_);
    const double s = h - delta_;
    return Bd + dBd * s + 0.5 * d2 * s * s;
  }
  double dBarrier(double h) const
  {
    if (h >= h_on_) return 0.0;
    if (h > delta_) return mu_ * (1.0 / h_on_ - 1.0 / h);
    return mu_ * (1.0 / h_on_ - 1.0 / delta_) + mu_ / (delta_ * delta_) * (h - delta_);
  }
  double d2Barrier(double h) const
  {
    if (h >= h_on_) return 0.0;
    if (h > delta_) return mu_ / (h * h);
    return mu_ / (delta_ * delta_);
  }

  Func f_;
  mutable decltype(f_.createData()) fdata_;
  double mu_, delta_, inv_scale_, h_on_;
  mutable VectorXs x_last_, u_last_;
  mutable bool valid_ = false, jac_ok_ = false;
};

using FovBarrierCost = RelaxedBarrierCost<FieldOfViewConstraint>;
using ManipBarrierCost = RelaxedBarrierCost<ManipulabilityConstraint>;

// List of allowed command line options
#define GETOPTARGS "cdhu:f:mvl"

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

void usage(const char *name, const char *badparam);
bool getOptions(int argc, const char **argv, bool &click_allowed, bool &display, std::string &urdf_path, std::string &camera_frame_name, bool &use_manipulability, bool &use_fov, bool &use_limits);

/*!

Print the program options.

  \param name : Program name.
  \param badparam : Bad parameter name.

*/
void usage(const char *name, const char *badparam)
{
  fprintf(stdout, "\n\
Tests a control law with the following characteristics:\n\
- eye-in-hand control\n\
- articular velocity are computed\n\
- servo on 4 points,\n\
- internal and external camera view displays.\n\
          \n\
SYNOPSIS\n\
  %s [-u <urdf_path>] [-f <camera_frame_name>] [-m] [-v] [-c] [-d] [-h]\n",
          name);

  fprintf(stdout, "\n\
OPTIONS:                                               Default\n\
  -u \n\
     Path towards the URDF of the robot.\n\
                  \n\
  -f \n\
     Name of the camera frame in the URDF.\n\
                  \n\
  -m \n\
     Turn off the manipulability constraints and cost.\n\
                  \n\
  -v \n\
     Turn off the field of view constraints and cost.\n\
                  \n\
  -l \n\
     Turn off the joints velocity and position limits constraints.\n\
                  \n\
  -c\n\
     Disable the mouse click. Useful to automate the \n\
     execution of this program without human intervention.\n\
                  \n\
  -d \n\
     Turn off the display.\n\
                  \n\
  -h\n\
     Print the help.\n");

  if (badparam)
    fprintf(stdout, "\nERROR: Bad parameter [%s]\n", badparam);
}
/*!

Set the program options.

  \param argc : Command line number of parameters.
  \param argv : Array of command line parameters.
  \param display : Display activation.
  \param click_allowed : Click activation.
  \param urdf_path : Path towards the URDF of the robot.
  \param camera_frame_name : Name of the camera frame in the URDF.
  \param use_manipulability : Dectivate manipulability constraints and cost.
  \param use_fov : Dectivate field of view constraints and cost.
  \param use_limits : Dectivate limits constraints and cost.

  \return false if the program has to be stopped, true otherwise.

*/
bool getOptions(int argc, const char **argv, bool &click_allowed, bool &display, std::string &urdf_path, std::string &camera_frame_name, bool &use_manipulability, bool &use_fov, bool &use_limits)
{
  const char *optarg_;
  int c;
  while ((c = vpParseArgv::parse(argc, argv, GETOPTARGS, &optarg_)) > 1) {

    switch (c) {
    case 'c':
      click_allowed = false;
      break;
    case 'd':
      display = false;
      break;
    case 'm':
      use_manipulability = false;
      break;
    case 'v':
      use_fov = false;
      break;
    case 'l':
      use_limits = false;
      break;
    case 'u':
      urdf_path = std::string(optarg_);
      break;
    case 'f':
      camera_frame_name = std::string(optarg_);
      break;
    case 'h':
      usage(argv[0], nullptr);
      return false;

    default:
      usage(argv[0], optarg_);
      return false;
    }
  }

  if ((c == 1) || (c == -1)) {
    // standalone param or error
    usage(argv[0], nullptr);
    std::cerr << "ERROR: " << std::endl;
    std::cerr << "  Bad argument " << optarg_ << std::endl << std::endl;
    return false;
  }

  return true;
}

int main(int argc, const char **argv)
{
#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
  std::shared_ptr<vpDisplay> displayInt;
#else
  vpDisplay *displayInt = nullptr;
#endif
  try {
    bool opt_click_allowed = true;
    bool opt_display = true;
    bool opt_use_manipulability = true;
    bool opt_use_fov = true;
    bool opt_use_limits = true;
    std::string opt_urdf_path = vpIoTools::createFilePath("urdf", "viper850_sim.urdf");
    std::string opt_camera_frame_name = std::string("camera_link");

    // Read the command line options
    if (getOptions(argc, argv, opt_click_allowed, opt_display, opt_urdf_path, opt_camera_frame_name, opt_use_manipulability, opt_use_fov, opt_use_limits) == false) {
      return EXIT_FAILURE;
    }

    // open a display for the visualization

    vpImage<unsigned char> Iint(480, 640, 255);

    if (opt_display) {
      // We open two displays, one for the internal camera view, the other one for
    // the external view
#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
      displayInt = vpDisplayFactory::createDisplay(Iint, 700, 0, "Internal view");
#else
      displayInt = vpDisplayFactory::allocateDisplay(Iint, 700, 0, "Internal view");
#endif
    }

    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << " Test program for MPC using aligator along ViSP " << std::endl;
    std::cout << " Eye-in-hand task control, articular velocity are computed" << std::endl;
    std::cout << " Simulation " << std::endl;
    std::cout << " task : servo 4 points " << std::endl;
    std::cout << " constraints:" << std::endl;
    if (opt_use_fov) {
      std::cout << "\t- Field of view" << std::endl;
    }
    if (opt_use_manipulability) {
      std::cout << "\t- Singularity avoidance" << std::endl;
    }
    if (opt_use_limits) {
      std::cout << "\t- Joints limits avoidance" << std::endl;
    }
    if ((!opt_use_limits) && (!opt_use_fov) && (!opt_use_manipulability)) {
      std::cout << "\tNONE" << std::endl;
    }
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    // sets the initial camera location
    vpHomogeneousMatrix cMo(-0.05, -0.05, 0.85, vpMath::rad(10), vpMath::rad(10), vpMath::rad(-30));

    // Desired pose
    vpHomogeneousMatrix cdMo(vpHomogeneousMatrix(0.0, 0.0, 0.8, vpMath::rad(0), vpMath::rad(0), vpMath::rad(0)));

    // Declaration of the robot
    vpSimulatorViper850 robot(opt_display);

    // Initialise the robot and especially the camera
    robot.init(vpViper850::TOOL_PTGREY_FLEA2_CAMERA, vpCameraParameters::perspectiveProjWithoutDistortion);
    robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);

    // Initialise the object for the display part
    robot.initScene(vpWireFrameSimulator::SQUARE_10CM, vpWireFrameSimulator::D_STANDARD);

    // Initialise the position of the object relative to the pose of the
    // robot's camera
    robot.initialiseObjectRelativeToCamera(cMo);

    // Set the desired position (for the display part)
    robot.setDesiredCameraPosition(cdMo);

    // Get the internal robot's camera parameters
    vpCameraParameters cam;
    robot.getCameraParameters(cam, Iint);

    // ---- Robot model --------------------------------------------------------
    pin::Model model;
    pin::urdf::buildModel(opt_urdf_path, model);
    const pin::FrameIndex cam_fid = model.getFrameId(opt_camera_frame_name);
    const int nq = model.nq, nu = model.nv;
    const double dt = 0.01;   // 50 Hz MPC
    const int horizon = 25;   // 0.25 s lookahead

    // ---- Parameters ---------------------------------------------------------

    // // ++++ For Cost terms ++++
    const double w_img_max = 5e3;     // Maxime image error weight
    const double w_img_min = 5e2; // Minimal image error weight
    double w_img = w_img_min;         // Current image error weight
    double w_img_term = 1e4;    // terminal image error weight
    const double w_pose_max = 5e1; // Maximum pose error weight
    const double w_pose_min = 5e1; // Minimum pose error weight
    double w_pose = 5e0; // Current pose error weight
    const double w_u_min = 2.5; // Minimal velocity regularization
    const double w_u_max = 500.; // Maximal velocity regularization
    double w_u = w_u_max;           // Current velocity regularization

    // // ++++ For Constraints terms ++++
    const double barrier_mu = 5e-2;     // barrier strength
    const double barrier_delta = 0.03;  // relaxation threshold (units of g: meters)
    const double manip_min = 0.03;  // minimum manipulability (tune! must be < initial w)
    const double manip_soft_thresh = 1.5 * manip_min;    // barrier starts to act around here
    const double manip_mu = 5e-2;
    const double manip_delta = 0.04;       // normalized: relax when w < 1.2 * manip_soft_thresh
    const double manip_h_on = 0.05;
    const double fov_margin = 0.9;  // keep the point within 90% of the half-image
    const double fov_h_on = 1e-3;

    // Camera intrinsics
    const double fx = cam.get_px(), fy = cam.get_py();
    const double img_w = Iint.getCols(), img_h = Iint.getRows();
    const double x_max = fov_margin * (img_w / 2.0) / fx;
    const double y_max = fov_margin * (img_h / 2.0) / fy;
    const double z_min = 0.05;      // minimum distance in front of the camera [m]

    if (opt_display) {
      // Get the internal view
      vpDisplay::display(Iint);
      robot.getInternalView(Iint);
      vpDisplay::flush(Iint);
    }

    // ---- Initial state ------------------------------------
    vpColVector q0_visp;
    robot.getPosition(vpRobot::ARTICULAR_FRAME, q0_visp);
    VectorXs q0;
    visp::visp2eigen(q0_visp, q0);
    vpHomogeneousMatrix e_M_c;
    robot.get_eMc(e_M_c);
    vpHomogeneousMatrix f_M_o = robot.get_fMo();

    // ---- Visual target ------------------------------------
    // sets the point coordinates in the object frame
    const unsigned int NB_POINTS = 4;
    std::vector<vpPoint> point(NB_POINTS);
    point[0].setWorldCoordinates(-0.05, -0.05, 0);
    point[3].setWorldCoordinates(-0.05, 0.05, 0);
    point[2].setWorldCoordinates(0.05, 0.05, 0);
    point[1].setWorldCoordinates(0.05, -0.05, 0);

    // sets the desired position of the feature point s*
    std::vector<vpFeaturePoint> pd(NB_POINTS);
    std::vector<vpImagePoint> pd_ip(NB_POINTS);

    std::shared_ptr<VisualTarget> target = std::make_shared<VisualTarget>();
    target->p_f.resize(NB_POINTS);
    target->s_ref.resize(NB_POINTS);
    for (unsigned int i = 0; i < NB_POINTS; ++i) {
      point[i].track(f_M_o);
      vpColVector fP = point[i].get_cP();
      vpColVector fP_3d = fP.extract(0, 3);
      Eigen::VectorXd fP_eigenX;
      visp::visp2eigen(fP_3d, fP_eigenX);
      target->p_f[i] = fP_eigenX;

      // Projection of the points
      point[i].track(cdMo);
      vpFeatureBuilder::create(pd[i], point[i]);
      target->s_ref[i][0] = pd[i].get_x();
      target->s_ref[i][1] = pd[i].get_y();
      vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), pd_ip[i]);
    }

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
    VectorXs umax = model.upperVelocityLimit;
    ControlBoxFunctionTpl<Scalar> u_box(nq, -umax, umax);


    ManipulabilityConstraint manip(model, cam_fid, manip_min, nu);        // hard
    ManipulabilityConstraint manip_soft(model, cam_fid, manip_soft_thresh, nu);  // for barrier
    FieldOfViewConstraint fov(model, cam_fid, target, x_max, y_max, z_min, nu);
    const double barrier_margin = 0.7;
    FieldOfViewConstraint fov_soft(model, cam_fid, target,
        barrier_margin * (img_w / 2.0) / fx,
        barrier_margin * (img_h / 2.0) / fy, z_min, nu);

    // ---- Cost ----------------------------------------------------------------
    ImagePointResidual img_res(model, cam_fid, target, nu);
    Eigen::Matrix4d cdMf_eigen;
    visp::visp2eigen(cdMo * f_M_o.inverse(), cdMf_eigen);
    PoseResidual pose_res(model, cam_fid, pin::SE3(cdMf_eigen), nu);
    ControlResidual ctrl_res(nq, nu);
    const MatrixXs R = w_u * MatrixXs::Identity(nu, nu);

    auto make_cost = [&](bool with_u) {
      CostStackTpl<Scalar> cs(space, nu);
      cs.addCost(WeightedResidualCost<ImagePointResidual>(
        space, img_res, w_img));
      cs.addCost(WeightedResidualCost<PoseResidual>(space, pose_res, w_pose));
      if (with_u)
        cs.addCost(WeightedResidualCost<ControlResidual>(space, ctrl_res, w_u));
      if (opt_use_fov) {
        cs.addCost(FovBarrierCost(space, fov_soft, barrier_mu, barrier_delta, fov_h_on));
      }
      if (opt_use_manipulability) {
        cs.addCost(ManipBarrierCost(space, manip_soft, manip_mu, manip_delta, manip_h_on, /*scale=*/manip_soft_thresh));
      }
      return cs;
      };

    // ---- Build the OCP --------------------------------------------------------
    using aligator::NegativeOrthantTpl;
    std::vector<xyz::polymorphic<StageModelTpl<Scalar>>> dummy; // (not needed)
    TrajOptProblemTpl<Scalar> problem(q0, nu, space, WeightedResidualCost<ImagePointResidual>(space, img_res, w_img_term));

    for (int t = 0; t < horizon; ++t) {
      StageModelTpl<Scalar> stage(make_cost(true), dyn);
      if (opt_use_limits) {
        stage.addConstraint(q_limits, NegativeOrthantTpl<Scalar>());
        stage.addConstraint(u_box, NegativeOrthantTpl<Scalar>());
      }
      if (opt_use_manipulability) {
        stage.addConstraint(manip, NegativeOrthantTpl<Scalar>());
      }
      if (opt_use_fov) {
        stage.addConstraint(fov, NegativeOrthantTpl<Scalar>());
      }
      problem.addStage(stage);
    }
    // Also constrain the terminal state
    if (opt_use_limits) {
      problem.addTerminalConstraint(q_limits, NegativeOrthantTpl<Scalar>());
    }
    if (opt_use_manipulability) {
      problem.addTerminalConstraint(manip, NegativeOrthantTpl<Scalar>());
    }
    if (opt_use_fov) {
      problem.addTerminalConstraint(fov, NegativeOrthantTpl<Scalar>());
    }

    // ---- Solver ----------------------------------------------------------------
    const double tol = 1e-3, mu_init = 1e-2;
    const std::size_t max_iters = 100;
    const double reg_init = 1e-6;
    SolverProxDDPTpl<Scalar> solver(tol, mu_init, max_iters, QUIET);
    solver.reg_init = reg_init;
    solver.setup(problem);

    std::vector<VectorXs> xs(horizon + 1, q0);
    std::vector<VectorXs> us(horizon, VectorXs::Zero(nu));

    unsigned int k = 0;
    // ---- MPC loop (simulation) ---------------------------------------------------
    VectorXs q = q0;
    vpColVector q_visp = q0_visp;
    pin::Data tmp(model);
    double s_sstart_inf_pix = std::numeric_limits<double>::max();
    const double thresh_error_pix = 0.005;
    const double max_iter = 20000; // 2000
    const unsigned int warm_up_iters = 0;
    bool clicked = false;
    while ((s_sstart_inf_pix > thresh_error_pix) && (k < max_iter) && (!clicked)) {
      ++k;
      vpColVector qdot;

      // Get the Time at the beginning of the loop
      double t = vpTime::measureTimeMs();

      // Update the robot configuration
      robot.getPosition(vpRobot::ARTICULAR_FRAME, q_visp);
      visp::visp2eigen(q_visp, q);
        // Get the current pose of the camera
      cMo = robot.get_cMo();

      if (k == 1) {
        std::cout << "Initial robot position with respect to the object frame:\n";
        cMo.print();
      }

      if (opt_display) {
        // Get the internal view and display it
        vpDisplay::display(Iint);
        robot.getInternalView(Iint);
        for (const auto &ip: pd_ip) {
          vpDisplay::displayCircle(Iint, ip, 2, vpColor::black, true);
        }
        vpDisplay::displayText(Iint, 20, 20, "Click to quit or wait until convergence.", vpColor::green);
        vpDisplay::flush(Iint);
        if (opt_display && opt_click_allowed) {
          clicked = vpDisplay::getClick(Iint, false);
        }
      }

      if (opt_display && opt_click_allowed && k == 1) {
        // suppressed for automate test
        std::cout << "Click in the internal view window to continue..." << std::endl;
        vpDisplay::getClick(Iint);
      }

      // 1) update the problem with current measurement
      problem.setInitState(q);          // older versions: problem.x0_init_ = q;
      // new points position
      f_M_o = robot.get_fMo(); //NB: this is constant, so the points never move with regard to the base frame of the robot
      for (unsigned int i = 0; i < NB_POINTS; i++) {
        point[i].track(f_M_o);
        // retrieve x,y and Z of the vpPoint structure
        vpColVector wP_visp = point[i].get_cP();
        vpColVector wP_3d = wP_visp.extract(0, 3);
        visp::visp2eigen(wP_3d, target->p_f[i]);
      }

      // Adaptation logic: adjust weights based on current error (s_sstart_inf_pix from previous iter)
      // High error -> Low w_img, High w_u (Smooth start)
      // Low error -> High w_img, Low w_u (Precise convergence)
      if (k > 1) {
        static const double alpha = 0.95;
        static const double one_minus_alpha = 1. - alpha;
        double target_w_img = std::max(w_img_min, w_img_max - 160.0 * s_sstart_inf_pix);
        w_img = alpha * w_img + one_minus_alpha * target_w_img;
        double target_w_u = std::min(w_u_max, w_u_min + 5.0 * s_sstart_inf_pix);
        w_u = alpha * w_u + one_minus_alpha * target_w_u;
      }

      // 2) solve, warm-started
      xs[0] = q;
      auto t0 = std::chrono::steady_clock::now();
      solver.run(problem, xs, us);
      auto t1 = std::chrono::steady_clock::now();
      const auto &res = solver.results_;
      VectorXs u0 = res.us[0];

      // Filter and limit the control input to avoid jitter, especially during non-convergence
      static VectorXs u_filtered = VectorXs::Zero(nu);
      double alpha_filter = 0.2; // base filter coefficient (1.0 = no filter)
      if (!res.conv) {
        alpha_filter = 0.05; // Stronger filter when not converged
      }
      u0 = alpha_filter * u0 + (1.0 - alpha_filter) * u_filtered;
      u_filtered = u0;

      // Safety clip to robot limits
      for (int i = 0; i < nu; ++i) {
        u0[i] = std::max(-umax[i], std::min(umax[i], u0[i]));
      }

      // 3) apply first control (here: integrate the simulated robot)
      visp::eigen2visp(u0, qdot);

      // 4) shift the solution for the next warm start
      for (int t = 0; t < horizon - 1; ++t) us[t] = res.us[t + 1];
      us[horizon - 1] = res.us[horizon - 1];
      for (int t = 0; t < horizon; ++t) xs[t] = res.xs[t + 1];
      xs[horizon] = res.xs[horizon];

      // send the camera velocity to the controller after MPC warm-up
      if (k >= warm_up_iters) {
        robot.setVelocity(vpRobot::ARTICULAR_FRAME, qdot);
      }

      ///TODO: Monitoring
      pin::forwardKinematics(model, tmp, q);
      pin::updateFramePlacement(model, tmp, cam_fid);

      double fov_slack = std::numeric_limits<double>::max();
      double s_sstart_inf = std::numeric_limits<double>::lowest();
      for (unsigned int i = 0; i < NB_POINTS; ++i) {
        const Eigen::Vector3d pc = tmp.oMf[cam_fid].actInv(target->p_f[i]);
        const double Z = pc.z();
        const double fov_slack_tmp = std::min({ x_max - pc.x() / Z, x_max + pc.x() / Z,
          y_max - pc.y() / Z, y_max + pc.y() / Z });
        fov_slack = std::min(fov_slack, fov_slack_tmp);
        const double s_sstar_inf_tmp = std::max({ std::abs(pc.x() / Z - target->s_ref[i][0]), std::abs(pc.y() / Z - target->s_ref[i][1]) });
        s_sstart_inf = std::max(s_sstar_inf_tmp, s_sstart_inf);
      }
      s_sstart_inf_pix = s_sstart_inf * fx;

      static bool once = true;
      if (once) {
        std::cout << "k" << ";solve_ms"<< ";iters="
          << ";conv"
          << ";qdot_0" << ";qdot_1"<< ";qdot_2"<< ";qdot_3"<< ";qdot_4"<< ";qdot_5"
          << ";w"
          << ";fov_slack"
          << ";w_img"
          << ";w_u"
          << ";||s - sstar||_{inf}(m)"
          << ";||s - sstar||_{inf}(pix)"
          << "\n";
        once = false;
      }
      std::stringstream ss;
      qdot.transpose().csvPrint(ss);
      std::string qdot_as_str = ss.str();
      std::replace(qdot_as_str.begin(), qdot_as_str.end(), ',', ';');
      qdot_as_str.erase(std::remove_if(qdot_as_str.begin(), qdot_as_str.end(), isspace), qdot_as_str.end());
      std::cout << k << ";" << std::chrono::duration<double, std::milli>(t1 - t0).count() << ";" << res.num_iters
        << ";" << res.conv
        << ";" << qdot_as_str
        << ";" << manip.manipulability(q)
        << ";" << fov_slack
        << ";" << w_img
        << ";" << w_u
        << ";" << s_sstart_inf
        << ";" << s_sstart_inf_pix
        << "\n";

      // The main loop has a duration of 10 ms at minimum
      vpTime::wait(t, 10);
    }
    vpColVector qdot(nu, 0.);
    robot.setVelocity(vpRobot::ARTICULAR_FRAME, qdot);

    ///TODO: Display task information
    auto debug_grads = [&](const VectorXs &q) {
      VectorXs u = VectorXs::Zero(nu);

      auto di = img_res.createData();
      img_res.evaluate(q, u, *di);
      img_res.computeJacobians(q, u, *di);
      VectorXs g_img = di->Jx_.transpose() * (w_img * di->value_);

      FovBarrierCost fb(space, fov_soft, barrier_mu, barrier_delta, fov_h_on);
      auto df = fb.createData();
      fb.evaluate(q, u, *df);
      fb.computeGradients(q, u, *df);

      ManipBarrierCost mb(space, manip_soft, manip_mu, manip_delta, manip_h_on, manip_soft_thresh);
      auto dm = mb.createData();
      mb.evaluate(q, u, *dm);
      mb.computeGradients(q, u, *dm);

      std::cout << "|g_img|=" << g_img.norm()
        << " |g_fov|=" << df->Lx_.norm()
        << " |g_manip|=" << dm->Lx_.norm()
        << " |sum|=" << (g_img + df->Lx_ + dm->Lx_).norm() << "\n";
      };

    // Update the robot configuration
    robot.getPosition(vpRobot::ARTICULAR_FRAME, q_visp);
    visp::visp2eigen(q_visp, q);
    debug_grads(q);

    // measured: from the simulator
    cMo = robot.get_cMo();
    double max_diff = 0;
    auto img_data = img_res.createData();
    img_res.evaluate(q, VectorXs::Zero(nu), *img_data);   // value = s_model - s*
    for (unsigned i = 0; i < NB_POINTS; ++i) {
      point[i].track(cMo);
      const double sx = point[i].get_x(), sy = point[i].get_y();
      // value = s_model - s* => s_model = value + s*
      const double mx = img_data->value_[2*i]   + target->s_ref[i].x();
      const double my = img_data->value_[2*i+1] + target->s_ref[i].y();
      max_diff = std::max({ max_diff, std::abs(mx - sx), std::abs(my - sy) });
    }
    std::cout << "|s_model - s_measured|_inf = " << max_diff
      << "  |e_model|_inf = " << img_data->value_.cwiseAbs().maxCoeff() << "\n";

    std::cout << "Final robot position with respect to the object frame:\n";
    // Get the current pose of the camera
    cMo = robot.get_cMo();
    cMo.print();
    std::cout << "\nFinal error cdMc:\n";
    // Get the current pose of the camera
    vpHomogeneousMatrix cdMc = cdMo * cMo.inverse();
    cdMc.print();
    std::cout << std::endl;

    if (opt_display && opt_click_allowed) {
      // suppressed for automate test
      std::cout << "Click in the internal view window to end..." << std::endl;
      vpDisplay::getClick(Iint);
    }
#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
    if (display != nullptr) {
      delete display;
  }
#endif
    return EXIT_SUCCESS;
}
  catch (const vpException &e) {
    std::cout << "Catch a ViSP exception: " << e << std::endl;
#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
    if (display != nullptr) {
      delete display;
  }
#endif
    return EXIT_FAILURE;
}
}
#elif !(defined(VISP_HAVE_DISPLAY))
int main()
{
  std::cout << "You do not have X11, or GDI (Graphical Device Interface) of OpenCV functionalities to display images..."
    << std::endl;
  std::cout << "Tip if you are on a unix-like system:" << std::endl;
  std::cout << "- Install X11, configure again ViSP using cmake and build again this example" << std::endl;
  std::cout << "Tip if you are on a windows-like system:" << std::endl;
  std::cout << "- Install GDI, configure again ViSP using cmake and build again this example" << std::endl;
  return EXIT_SUCCESS;
}
#elif !(defined(VISP_HAVE_LAPACK) || defined(VISP_HAVE_EIGEN3) || defined(VISP_HAVE_OPENCV))
int main()
{
  std::cout << "Cannot run this example: install Lapack, Eigen3 or OpenCV" << std::endl;
  return EXIT_SUCCESS;
}
#else
int main()
{
  std::cout << "You do not have threading capabilities" << std::endl;
  std::cout << "Tip:" << std::endl;
  std::cout << "- Install pthread, configure again ViSP using cmake and build again this example" << std::endl;
  return EXIT_SUCCESS;
}
#endif
