#ifndef COLMAP_NORMAL_EQUATIONS_H
#define COLMAP_NORMAL_EQUATIONS_H

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <string>

namespace colmap {

// Normal equations for solving least squares problems JtJ and JtR are always
// updated on-the-fly, the full Jacobian is never stored in memory.
template <int N>
class NormalEquations {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Constructor for creating empty normal equations.
  NormalEquations();

  /// Destructor.
  ~NormalEquations();

  /// Reset normal equations.
  void Reset();

  /**
   * @brief   Update normal equations with a new residual and its
   *          respective weight and Jacobian.
   *          JtJ and JtR are updated directly on-the-fly.
   * @param   jacobian_row    Jacobian row for the residual.
   * @param   residual        Residual.
   * @param   weight          Weight of the residual/Jacobian.
   */
  void Update(const Eigen::Matrix<float, N, 1>& jacobian_row,
              const float residual, const float weight = 1.0f);

  /**
   * @brief   Solve normal equations using Cholesky LDLT decomposition.
   * @return  Solution, i.e. parameter update.
   */
  Eigen::Matrix<float, N, 1> Solve() const;

  /// Get overall cost (computed from added residuals and their weights).
  double Error() const;

  /// Get average per-residual cost.
  double AverageError() const;

 protected:
  Eigen::Matrix<float, N, N> JtJ_;
  Eigen::Matrix<float, N, 1> JtR_;
  double error_;
  size_t num_residuals_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int N>
NormalEquations<N>::NormalEquations()
    : JtJ_(Eigen::Matrix<float, N, N>::Zero()),
      JtR_(Eigen::Matrix<float, N, 1>::Zero()),
      error_(0.0),
      num_residuals_(0) {}

template <int N>
NormalEquations<N>::~NormalEquations() = default;

template <int N>
void NormalEquations<N>::Reset() {
  JtJ_.setZero();
  JtR_.setZero();
  error_ = 0.0;
  num_residuals_ = 0;
}

template <int N>
void NormalEquations<N>::Update(const Eigen::Matrix<float, N, 1>& jacobian_row,
                                const float residual, const float weight) {
  for (int k = 0; k < N; ++k) {
    // update A = Jt*J (including robust weights)
    for (int j = 0; j < N; ++j)
      JtJ_(k, j) += jacobian_row[j] * weight * jacobian_row[k];

    // update b = Jt*r (apply robust weights)
    JtR_[k] += jacobian_row[k] * weight * residual;
  }

  // update overall error
  error_ += static_cast<double>(residual * weight * residual);

  // update number of added residuals
  ++num_residuals_;
}

template <int N>
Eigen::Matrix<float, N, 1> NormalEquations<N>::Solve() const {
  // solve normal equations using Cholesky LDLT decomposition
  return -(JtJ_.ldlt().solve(JtR_));
}

template <int N>
double NormalEquations<N>::Error() const {
  return error_;
}

template <int N>
double NormalEquations<N>::AverageError() const {
  // calculate and return final overall error
  if (num_residuals_ > 0)
    return error_ / static_cast<double>(num_residuals_);
  else
    return 0.0;
}

}  // namespace colmap

#endif  // COLMAP_NORMAL_EQUATIONS_H
