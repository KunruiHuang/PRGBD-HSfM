#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unordered_map>

#include "estimators/two_view_geometry.h"
#include "util/hash.h"
#include "util/math.h"
#include "util/types.h"

namespace colmap {

// Ref: Efficient and Large Scale Rotation Averaging" by Chatterjee and Govindu
// (ICCV 2013)
class RobustRotationEstimator {
 public:
  struct Options {
    // Maximum number of times to run L1 minimization. L1 is very slow (compared
    // to L2), but is very robust to outliers. Typically only a few iterations
    // are needed in order for the solution to reside within the cone of
    // convergence for L2 solving.
    int max_num_l1_iterations = 5;

    // Average step size threshold to terminate the L1 minimization
    double l1_step_convergence_threshold = 0.001;

    // The number of iterative reweighted least squares iterations to perform.
    int max_num_irls_iterations = 100;

    // Average step size threshold to termininate the IRLS minimization
    double irls_step_convergence_threshold = 0.001;

    // This is the point where the Huber-like cost function switches from L1 to
    // L2.
    double irls_loss_parameter_sigma = DegToRad(5.0);
  };

  explicit RobustRotationEstimator(const Options& options);

  bool EstimateRotations(
      const std::unordered_map<image_pair_t, std::vector<TwoViewGeometry>>&
          view_pairs,
      std::unordered_map<image_t, Eigen::Vector3d>* global_orientations);

  void AddRelativeRotationConstraint(const image_pair_t view_id_pair,
                                     const Eigen::Vector3d& relative_rotation);

  bool EstimateRotations(
      std::unordered_map<image_t, Eigen::Vector3d>* global_orientations);

 private:
  void SetupLinearSystem();

  bool SolveL1Regression();

  bool SolveIRLS();

  void UpdateGlobalRotations();

  void ComputeResiduals();

  double ComputeAverageStepSize();

  void NormalizedWeights();

  bool RefineGlobalOrientations();

  static const int kConstantRotationIndex = -1;

  const Options options_;

  std::vector<double> relative_rotations_weights_;
  std::vector<std::pair<image_pair_t, Eigen::Vector3d>> relative_rotations_;

  std::unordered_map<image_t, Eigen::Vector3d>* global_orientations_;

  std::unordered_map<image_t, int> view_id_to_index_;

  // Ax = b
  Eigen::SparseMatrix<double> sparse_matrix_;

  // x
  Eigen::VectorXd tangent_space_step_;

  // b
  Eigen::VectorXd tangent_space_residual_;
};

}  // namespace colmap
