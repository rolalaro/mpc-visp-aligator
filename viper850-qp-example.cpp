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
#include <visp3/vs/vpServoDisplay.h>

#include <proxsuite/proxqp/dense/dense.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <chrono>
#include <iostream>
#include <memory>

namespace pin = pinocchio;
using Scalar = double;
using VectorXs = Eigen::VectorXd;
using MatrixXs = Eigen::MatrixXd;
using ConstVectorRef = Eigen::Ref<const VectorXs>;

// List of allowed command line options
#define GETOPTARGS "cdhu:f:g:mvl"

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

// ---------------------------------------------------------------------------
// Compute the manipulability and its gradient
// ---------------------------------------------------------------------------
class Manipulability
{
public:
  Manipulability(const pin::Model &model, pin::FrameIndex ee_frame)
    : model_(model), pdata_(model), fid_(ee_frame),
    J_(6, model.nv)
  {
    J_.setZero();
  }

  double manipulability(const VectorXs &q) const
  {
    pin::computeFrameJacobian(model_, pdata_, q, fid_, pin::LOCAL, J_);
    const double det = (J_ * J_.transpose()).determinant();
    return std::sqrt(std::max(det, 0.0));
  }

  MatrixXs gradient(const ConstVectorRef &q) const
  {
    const double eps = 1e-6;
    MatrixXs grad(1, model_.nv);
    for (int i = 0; i < model_.nv; ++i) {
      VectorXs qp = q, qm = q;
      qp[i] += eps;
      qm[i] -= eps;
      grad(0, i) = (manipulability(qp) - manipulability(qm)) / (2 * eps);
    }
    return grad;
  }

private:
  const pin::Model &model_;
  mutable pin::Data pdata_;
  pin::FrameIndex fid_;
  mutable MatrixXs J_;
};

void usage(const char *name, const char *badparam);
bool getOptions(int argc, const char **argv, bool &click_allowed, bool &display, std::string &urdf_path, std::string &camera_frame_name, bool &use_manipulability, bool &use_fov, bool &use_limits, double &gain);

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
  %s [-u <urdf_path>] [-f <camera_frame_name>] [-g gain] [-m] [-v] [-c] [-d] [-h]\n",
          name);

  fprintf(stdout, "\n\
OPTIONS:                                               Default\n\
  -u <path_to_urdf>\n\
     Path towards the URDF of the robot.\n\
                  \n\
  -f <camera_frame_name>\n\
     Name of the camera frame in the URDF.\n\
                  \n\
  -g <gain> \n\
     Gain of the control law. \n\
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
bool getOptions(int argc, const char **argv, bool &click_allowed, bool &display, std::string &urdf_path, std::string &camera_frame_name, bool &use_manipulability, bool &use_fov, bool &use_limits, double &gain)
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
    case 'g':
      gain = std::atof(optarg_);
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
    double opt_gain = 1.;

    // Read the command line options
    if (getOptions(argc, argv, opt_click_allowed, opt_display, opt_urdf_path, opt_camera_frame_name, opt_use_manipulability, opt_use_fov, opt_use_limits, opt_gain) == false) {
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
    vpHomogeneousMatrix cMo(-0.05, -0.05, 0.8, vpMath::rad(10), vpMath::rad(10), vpMath::rad(-30));

    // Desired pose
    vpHomogeneousMatrix cdMo(vpHomogeneousMatrix(0.0, 0.0, 0.8, vpMath::rad(0), vpMath::rad(0), vpMath::rad(0)));

    // Declaration of the robot
    vpSimulatorViper850 robot(opt_display);

    // Initialise the robot and especially the camera
    robot.init(vpViper850::TOOL_PTGREY_FLEA2_CAMERA, vpCameraParameters::perspectiveProjWithoutDistortion);
    robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);

    // Initialise the object for the display part
    robot.initScene(vpWireFrameSimulator::PLATE, vpWireFrameSimulator::D_STANDARD);

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
    const double dt_ms = 10.; // 100 Hz loop
    const double dt = dt_ms / 1000.;

    // ---- Parameters ---------------------------------------------------------

    // Camera intrinsics
    // const double fx = cam.get_px(), fy = cam.get_py();
    // const double img_w = Iint.getCols(), img_h = Iint.getRows();
    // const double x_max = fov_margin * (img_w / 2.0) / fx;
    // const double y_max = fov_margin * (img_h / 2.0) / fy;
    // const double z_min = 0.05;      // minimum distance in front of the camera [m]

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

    // ---- Preparing the visual servoing ----------------------------------------------------------
    vpServo task; // To compute the error and the interaction matrix

    // define the task
    // - we want an eye-in-hand control law
    // - articular velocity are computed
    task.setServo(vpServo::EYEINHAND_L_cVe_eJe);
    task.setInteractionMatrixType(vpServo::MEAN);

    // Set the position of the end-effector frame in the camera frame
    vpVelocityTwistMatrix cVe(e_M_c.inverse());
    task.set_cVe(cVe);

    // Set the task gain
    task.setLambda(opt_gain);

    // Set the Jacobian (expressed in the end-effector frame)
    vpMatrix eJe;
    robot.get_eJe(eJe);
    task.set_eJe(eJe);

    // sets the point coordinates in the object frame
    const unsigned int NB_POINTS = 4;
    std::vector<vpPoint> point(NB_POINTS);
    point[0].setWorldCoordinates(-0.1, -0.1, 0);
    point[3].setWorldCoordinates(0.1, -0.1, 0);
    point[2].setWorldCoordinates(0.1, 0.1, 0);
    point[1].setWorldCoordinates(-0.1, 0.1, 0);

    // sets the desired position of the feature point s*
    std::vector<vpFeaturePoint> p(NB_POINTS);
    std::vector<vpFeaturePoint> pd(NB_POINTS);
    std::vector<vpImagePoint> pd_ip(NB_POINTS);

    for (unsigned int i = 0; i < NB_POINTS; ++i) {
      point[i].track(cMo);
      vpFeatureBuilder::create(p[i], point[i]);

      // Projection of the points
      point[i].track(cdMo);
      vpFeatureBuilder::create(pd[i], point[i]);
      vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), pd_ip[i]);

      // Adding the feature to the visual servoing
      task.addFeature(p[i], pd[i]);
    }

    // Display task information
    task.print();

    // ---- Preparing the QP problem ----------------------------------------------------------
    // ---- Constraints ----
    // q limits:  [I; -I] q + [-qmax; qmin] <= 0
    MatrixXs Aq(2 * nq, nq);
    Aq << MatrixXs::Identity(nq, nq), -MatrixXs::Identity(nq, nq);
    VectorXs cq(2 * nq);
    cq << -model.upperPositionLimit, model.lowerPositionLimit;

    // velocity box
    VectorXs umax = model.upperVelocityLimit;
    VectorXs qmin = model.lowerPositionLimit;
    VectorXs qmax = model.upperPositionLimit;
    // 4. Box Constraints: Physical Joint Velocity Limits
    const double damper_gain = 0.1;
    const double eta = 0.1 * (qmax.maxCoeff() - qmin.minCoeff());
    const double damper_gain_over_eta = damper_gain / eta;
    VectorXs q_dot_min = -1. * umax; // rad/s
    VectorXs q_dot_max = umax; // rad/s
    std::cout << "Velocity limits:" << std::endl << "\tmin: " << q_dot_min.transpose() << "\n\tmax: " << q_dot_max.transpose() << std::endl;

    // generate a QP problem
    Manipulability manip_cpter(model, cam_fid);
    const double weight_slack = 1e3; // Heavy penalty for breaking the soft manipulability constraint (slack)
    const double cost_manipulability = 1e-2;  // Weight for smooth null-space posture projection
    const double manip_beta = 0.1; // Step sizing for manipulability optimization step
    const double manip_thresh = manip_cpter.manipulability(q0) / 2.;  // Safety threshold floor for manipulability

    proxsuite::proxqp::dense::isize dim = eJe.getCols(); // We want to find q, which is of size n
    proxsuite::proxqp::dense::isize n_eq(0); // For the moment no equality constraints
    proxsuite::proxqp::dense::isize n_in(0); // For the moment no inequality constraints, but we could add for instance joint limits avoidance
    if (opt_use_manipulability) {
      ++dim; // We also have a slack variable in the state
      n_in = 2; // two inequalities: one for the manipulability and one for the slack variable
    }
    // we generate a qp, so the function used from helpers.hpp is
    // in proxqp namespace. The qp is in dense eigen format and
    // you can control its sparsity ratio and strong convexity factor.
    proxsuite::proxqp::dense::Model<double> qp_model(dim, n_eq, n_in, opt_use_limits);

    // load PROXQP solver with dense backend
    proxsuite::proxqp::dense::QP<double> qp(dim, n_eq, n_in, opt_use_limits);

    // ---- Main loop ----
    unsigned int iter = 0;
    task.computeControlLaw();
    double squared_error = (task.getError()).sumSquare();
    bool clicked = false;
    VectorXs q_eigen(nq);
    // loop
    while ((squared_error > 1e-6) && (!clicked)) {
      std::cout << "---------------------------------------------" << iter++ << std::endl;
      vpColVector q;
      robot.getPosition(vpRobot::ARTICULAR_FRAME, q);

      // Set the Jacobian (expressed in the end-effector frame)
      // since q is modified eJe is modified
      robot.get_eJe(eJe);
      task.set_eJe(eJe);

      // get the robot position
      vpHomogeneousMatrix f_M_c = robot.get_fMc(q);
      // Compute the position of the object frame in the camera frame
      cMo = f_M_c.inverse() * f_M_o;

      // update new point position and corresponding features
      for (unsigned i = 0; i < 4; i++) {
        point[i].track(cMo);
        // retrieve x,y and Z of the vpPoint structure
        vpFeatureBuilder::create(p[i], point[i]);
      }

      if (opt_display) {
        vpDisplay::display(Iint);
        robot.getInternalView(Iint);
        vpDisplay::displayText(Iint, 20, 20, "Click to quit or wait until convergence.", vpColor::green);
        vpServoDisplay::display(task, cam, Iint);
        // externalview.display(Iext, cextMo, cMo, cam, vpColor::green);
        vpDisplay::flush(Iint);
        // vpDisplay::flush(Iext);
        clicked = vpDisplay::getClick(Iint, false);
      }

      // compute the control law
      task.computeError();
      task.computeInteractionMatrix();
      vpMatrix Jtask = task.getInteractionMatrix() * cVe * eJe;
      vpMatrix JtaskT = Jtask.transpose();
      vpMatrix JtastT_Jtask = JtaskT * Jtask;
      const double alpha = 1e-4;
      vpMatrix I;
      I.eye(JtastT_Jtask.getRows());
      vpMatrix H = JtastT_Jtask + alpha * I;
      vpColVector e = task.getError();
      if (opt_use_manipulability) {
        static bool once = true;
        if (once) {
          qp_model.H.resize(nu+1, nu+1);
          qp_model.g.resize(nu+1);
          qp_model.g(nu) = 0.;

          qp_model.C.resize(2, nu+1);
          qp_model.C(0, nu) = -1.;
          qp_model.C.block(nu, 0, 1, nu) = VectorXs::Zero(1, nu);
          qp_model.C(nu, nu) = -1.;

          qp_model.u = VectorXs::Zero(2, 1);
          once = false;
        }
        // Managing the visual servoing part of H
        MatrixXs H_visual(nu, nu);
        visp::visp2eigen(H, H_visual);
        qp_model.H.block(0, 0, nu, nu) = H_visual;
        // Managing the slack part of H
        qp_model.H(nu, nu) = weight_slack;
        qp_model.H *= 2.;

        // Managing the visual servoing part of g
        MatrixXs g_visual(nu, 1);
        visp::visp2eigen(task.lambda.getLastValue() * JtaskT * e, g_visual);
        visp::visp2eigen(q, q_eigen);
        // Managing the manipulability part of g
        MatrixXs grad_manip = manip_cpter.gradient(q_eigen);
        // Summing them up
        qp_model.g.block(0, 0, nu, 1) = g_visual - cost_manipulability * manip_beta * grad_manip;

        // Manipulability soft constraints
        qp_model.C.block(0, 0, 1, nu) = grad_manip.transpose();
        qp_model.u(0, 0) = manip_cpter.manipulability(q_eigen) - manip_thresh;
      }
      else {
        visp::visp2eigen(H, qp_model.H);
        visp::visp2eigen(task.lambda.getLastValue() * JtaskT * e, qp_model.g);
      }

      if (opt_use_limits) {
        VectorXs q_eigen;
        visp::visp2eigen(q, q_eigen);
        // Handling simultaneously joint velocity limits and joints position limits
        // using the velocity damper method
        qp_model.l_box.cwiseMax(damper_gain_over_eta * (qmin - q_eigen));
        qp_model.u_box.cwiseMin(damper_gain_over_eta * (qmax - q_eigen));
        qp.init(qp_model.H,
            qp_model.g,
            qp_model.A,
            qp_model.b,
            qp_model.C,
            qp_model.l,
            qp_model.u,
            qp_model.l_box,
            qp_model.u_box);
      }
      else {
        qp.init(qp_model.H,
            qp_model.g,
            qp_model.A,
            qp_model.b,
            qp_model.C,
            qp_model.l,
            qp_model.u);
      }
      qp.solve();
      auto state_eigen = qp.results.x;
      VectorXs q_eigen = state_eigen.block(0, 0, nu, 1);
      visp::eigen2visp(q_eigen, q);

      // send the camera velocity to the controller
      robot.setVelocity(vpRobot::ARTICULAR_FRAME, q);
      squared_error = (task.getError()).sumSquare();
      std::cout << "|| s - s* || = " << squared_error << "\tslack = " << (opt_use_manipulability ? std::to_string(state_eigen(nu, 0)) : std::string("NaN")) << std::endl;

      vpTime::wait(dt_ms);
    }
    vpColVector qdot_null(nq, 0.);
    // Stop the robot
    robot.setVelocity(vpRobot::ARTICULAR_FRAME, qdot_null);

    // Display task information
    task.print();

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
