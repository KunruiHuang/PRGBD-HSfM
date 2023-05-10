#include "sfm/global/robust_rotation_estimator.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "base/cost_functions.h"
#include "base/database.h"
#include "base/pose.h"
#include "math/l1_solver.h"
#include "math/matrix/sparse_cholesky_llt.h"
#include "optim/bundle_adjustment.h"
#include "util/hash.h"
#include "util/map_util.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/types.h"

namespace colmap {

RobustRotationEstimator::RobustRotationEstimator(const Options& options)
    : options_(options), global_orientations_(nullptr) {}

bool RobustRotationEstimator::EstimateRotations(
    const std::unordered_map<image_pair_t, std::vector<TwoViewGeometry>>&
        view_pairs,
    std::unordered_map<image_t, Eigen::Vector3d>* global_orientations) {
  for (const auto& view_pair : view_pairs) {
    for (const auto& edge : view_pair.second) {
      Eigen::Vector3d rji;
      ceres::QuaternionToAngleAxis(edge.qvec.data(), rji.data());
      AddRelativeRotationConstraint(view_pair.first, rji);
    }
  }

  return EstimateRotations(global_orientations);
}

void RobustRotationEstimator::AddRelativeRotationConstraint(
    const image_pair_t view_id_pair, const Eigen::Vector3d& relative_rotation) {
  relative_rotations_.emplace_back(view_id_pair, relative_rotation);
}

bool RobustRotationEstimator::EstimateRotations(
    std::unordered_map<image_t, Eigen::Vector3d>* global_orientations) {
  CHECK_GT(relative_rotations_.size(), 0)
      << "Relative rotation constraints must be added to the robust rotation "
         "solver before estimating global rotations.";
  global_orientations_ = CHECK_NOTNULL(global_orientations);

  PrintHeading1("Estimate Global Rotations");
  Timer timer;
  timer.Start();

  int index = -1;
  view_id_to_index_.reserve(global_orientations_->size());
  for (const auto& orientation : *global_orientations_) {
    view_id_to_index_[orientation.first] = index;
    ++index;
  }

  std::cout << "Setup sparse matrix..." << std::flush;

  Eigen::SparseMatrix<double> sparse_mat;
  SetupLinearSystem();

  std::cout << StringPrintf(" %d in %.3fs", view_id_to_index_.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  std::cout << "L1 regression..." << std::flush;
  timer.Restart();
  // L1 minimization
  if (!SolveL1Regression()) {
    std::cout << "ERROR: Could not solve the L1 regression step.";
    return false;
  }

  std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;

  PrintHeading2("L2 IRLS...");
  timer.Restart();
  // Refine based on L2 IRLS
  if (!SolveIRLS()) {
    std::cout << "ERROR: Could not solve the least squares error step.";
    return false;
  }
  std::cout << StringPrintf("L2 IRLS elapsed %.3fs", timer.ElapsedSeconds())
            << std::endl;

  // Refine based on ceres solver
  PrintHeading2("Refine global rotations.");
  if (!RefineGlobalOrientations()) {
    std::cout << "ERROR: Cloud not refine the global orientations step.";
    return false;
  }

  return true;
}

void RobustRotationEstimator::SetupLinearSystem() {
  tangent_space_step_.resize((global_orientations_->size() - 1) * 3);
  tangent_space_residual_.resize(relative_rotations_.size() * 3);
  sparse_matrix_.resize(relative_rotations_.size() * 3,
                        (global_orientations_->size() - 1) * 3);

  int rotation_error_index = 0;
  std::vector<Eigen::Triplet<double>> triplet_list;
  for (const auto& relative_rotation : relative_rotations_) {
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(relative_rotation.first, &image_id1,
                                &image_id2);
    const int view1_index = FindOrDie(view_id_to_index_, image_id1);
    if (view1_index != kConstantRotationIndex) {
      triplet_list.emplace_back(3 * rotation_error_index, 3 * view1_index,
                                -1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 1,
                                3 * view1_index + 1, -1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 2,
                                3 * view1_index + 2, -1.0);
    }

    const int view2_index = FindOrDie(view_id_to_index_, image_id2);
    if (view2_index != kConstantRotationIndex) {
      triplet_list.emplace_back(3 * rotation_error_index + 0,
                                3 * view2_index + 0, 1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 1,
                                3 * view2_index + 1, 1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 2,
                                3 * view2_index + 2, 1.0);
    }

    ++rotation_error_index;
  }
  sparse_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

bool RobustRotationEstimator::SolveL1Regression() {
  L1Solver<Eigen::SparseMatrix<double>>::Options options;
  options.max_num_iterations = 5;
  L1Solver<Eigen::SparseMatrix<double>> l1_solver(options, sparse_matrix_);

  tangent_space_step_.setZero();
  ComputeResiduals();
  for (int i = 0; i < options_.max_num_l1_iterations; i++) {
    l1_solver.Solve(tangent_space_residual_, &tangent_space_step_);
    UpdateGlobalRotations();
    ComputeResiduals();

    double avg_step_size = ComputeAverageStepSize();

    if (avg_step_size <= options_.l1_step_convergence_threshold) {
      break;
    }
    options.max_num_iterations *= 2;
    l1_solver.SetMaxIterations(options.max_num_iterations);
  }
  return true;
}

bool RobustRotationEstimator::SolveIRLS() {
  const int num_edges = tangent_space_residual_.size() / 3;

  // Set up the linear solver and analyze the sparsity pattern of the
  // system. Since the sparsity pattern will not change with each linear solve
  // this can help speed up the solution time.
  SparseCholeskyLLt linear_solver;
  linear_solver.AnalyzePattern(sparse_matrix_.transpose() * sparse_matrix_);
  if (linear_solver.Info() != Eigen::Success) {
    LOG(ERROR) << "Cholesky decomposition failed.";
    return false;
  }

  std::cout << "Iteration   SqError         Delta" << std::endl;
  const std::string row_format = "  % 4d     % 4.4e     % 4.4e";

  ComputeResiduals();

  Eigen::ArrayXd weights(num_edges * 3);
  Eigen::SparseMatrix<double> at_weight;
  relative_rotations_weights_.resize(num_edges);
  for (int i = 0; i < options_.max_num_irls_iterations; i++) {
    // Compute the Huber-like weights for each error term.
    const double& sigma = options_.irls_loss_parameter_sigma;
    for (int k = 0; k < num_edges; ++k) {
      double e_sq = tangent_space_residual_.segment<3>(3 * k).squaredNorm();
      double tmp = e_sq + sigma * sigma;
      double w = sigma / (tmp * tmp);
      weights.segment<3>(3 * k).setConstant(w);
      relative_rotations_weights_[k] = w;
    }

    // Update the factorization for the weighted values.
    at_weight = sparse_matrix_.transpose() * weights.matrix().asDiagonal();
    linear_solver.Factorize(at_weight * sparse_matrix_);
    if (linear_solver.Info() != Eigen::Success) {
      std::cout << "ERROR: Failed to factorize the least squares system.";
      return false;
    }

    // Solve the least squares problem..
    tangent_space_step_ =
        linear_solver.Solve(at_weight * tangent_space_residual_);
    if (linear_solver.Info() != Eigen::Success) {
      std::cout << "ERROR: Failed to solve the least squares system.";
      return false;
    }

    UpdateGlobalRotations();
    ComputeResiduals();
    const double avg_step_size = ComputeAverageStepSize();

    std::cout << StringPrintf(row_format.c_str(), i,
                              tangent_space_residual_.squaredNorm(),
                              avg_step_size)
              << std::endl;

    if (avg_step_size < options_.irls_step_convergence_threshold) {
      std::cout << "IRLS Converged in " << i + 1 << " iterations." << std::endl;
      break;
    }
  }

  return true;
}

void RobustRotationEstimator::UpdateGlobalRotations() {
  for (auto& rotation : *global_orientations_) {
    const int view_index = FindOrDie(view_id_to_index_, rotation.first);
    if (view_index == kConstantRotationIndex) {
      continue;
    }

    // Apply the rotation change to the global orientation.
    const Eigen::Vector3d& rotation_change =
        tangent_space_step_.segment<3>(3 * view_index);
    rotation.second = MultiplyRotations(rotation.second, rotation_change);
  }
}

void RobustRotationEstimator::ComputeResiduals() {
  int rotation_error_index = 0;
  for (const auto& relative_rotation : relative_rotations_) {
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(relative_rotation.first, &image_id1,
                                &image_id2);
    const Eigen::Vector3d& relative_rotation_aa = relative_rotation.second;
    const Eigen::Vector3d& rotation1 =
        FindOrDie(*global_orientations_, image_id1);
    const Eigen::Vector3d& rotation2 =
        FindOrDie(*global_orientations_, image_id2);
    tangent_space_residual_.segment<3>(3 * rotation_error_index) =
        MultiplyRotations(-rotation2,
                          MultiplyRotations(relative_rotation_aa, rotation1));
    ++rotation_error_index;
  }
}

double RobustRotationEstimator::ComputeAverageStepSize() {
  // compute the average step size of the update in tangent_space_step_
  const int numVertices = tangent_space_step_.size() / 3;
  double delta_V = 0;
  for (int k = 0; k < numVertices; ++k) {
    delta_V += tangent_space_step_.segment<3>(3 * k).norm();
  }
  return delta_V / numVertices;
}

void RobustRotationEstimator::NormalizedWeights() {
  double max_weight = std::numeric_limits<double>::min();
  for (const auto& weight : relative_rotations_weights_) {
    if (weight > max_weight) {
      max_weight = weight;
    }
  }

  // normalized to [0, 1]
  for (auto& weight : relative_rotations_weights_) {
    weight /= max_weight;
  }
}

bool RobustRotationEstimator::RefineGlobalOrientations() {
  // normalize weight improved numerical stability
  NormalizedWeights();
  CHECK_EQ(relative_rotations_.size(), relative_rotations_weights_.size());

  ceres::Problem problem;
  const double robust_loss_thres = 0.03;  // 2°
  for (size_t i = 0; i < relative_rotations_.size(); ++i) {
    const auto& relative_rotation = relative_rotations_[i];
    const auto weight = relative_rotations_weights_[i];
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(relative_rotation.first, &image_id1,
                                &image_id2);
    Eigen::Vector3d& image1_orientation = global_orientations_->at(image_id1);
    Eigen::Vector3d& image2_orientation = global_orientations_->at(image_id2);
    ceres::CostFunction* cost_function =
        RelativeRotationCostFunction::Create(relative_rotation.second, weight);
    ceres::LossFunction* loss_function =
        new ceres::CauchyLoss(robust_loss_thres * weight);

    problem.AddResidualBlock(cost_function, loss_function,
                             image1_orientation.data(),
                             image2_orientation.data());
  }

  ceres::Solver::Options solverOptions;
  solverOptions.minimizer_progress_to_stdout = true;
  solverOptions.logging_type = ceres::SILENT;
  if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE)) {
    solverOptions.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    solverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  } else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(
                 ceres::CX_SPARSE)) {
    solverOptions.sparse_linear_algebra_library_type = ceres::CX_SPARSE;
    solverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  } else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(
                 ceres::EIGEN_SPARSE)) {
    solverOptions.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    solverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  } else {
    solverOptions.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  }

  solverOptions.num_threads = GetEffectiveNumThreads(-1);
  ceres::Solver::Summary summary;
  ceres::Solve(solverOptions, &problem, &summary);

  PrintGolbalRotationSolverSummary(summary);

  return summary.IsSolutionUsable();
}

}  // namespace colmap