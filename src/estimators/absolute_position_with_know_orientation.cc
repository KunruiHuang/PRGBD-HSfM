
#include "estimators/absolute_position_with_know_orientation.h"

#include <glog/logging.h>

namespace colmap {

std::vector<AbsolutePositionWithKnowOrientationEstimator::M_t>
AbsolutePositionWithKnowOrientationEstimator::Estimate(
    const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D) {
  CHECK_EQ(points2D.size(), 2);
  CHECK_EQ(points3D.size(), 2);

  // 利用重投影误差约束估计相机的绝对位置, 构建线性方程
  // [-1  0  u] * c = [x - u * z]
  // [0  -1  v]     = [y - v * z]
  Eigen::Matrix<double, 4, 3> A;
  A.block<2, 2>(0, 0) = -1.0 * Eigen::Matrix2d::Identity();
  A.block<2, 2>(2, 0) = -1.0 * Eigen::Matrix2d::Identity();
  A.block<2, 1>(0, 2) = points2D[0];
  A.block<2, 1>(2, 2) = points2D[1];

  Eigen::Vector4d b;
  b.head<2>() = points3D[0].head<2>() - points2D[0] * points3D[0].z();
  b.tail<2>() = points3D[1].head<2>() - points2D[1] * points3D[1].z();

  Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 4, 3>> linear_solver(A);
  // 检查线性方程的秩是否满足
  if (linear_solver.rank() != 3) {
    return {};
  }

  std::vector<M_t> models;
  models.emplace_back(linear_solver.solve(b));
  return models;
}

void AbsolutePositionWithKnowOrientationEstimator::Residuals(
    const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D,
    const M_t& position, std::vector<double>* residuals) {
  // 计算重投影误差的平方
  CHECK_EQ(points2D.size(), points3D.size());

  residuals->resize(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    const Eigen::Vector2d reprojected_feature =
        (points3D[i] + position).hnormalized();
    (*residuals)[i] = (points2D[i] - reprojected_feature).squaredNorm();
  }
}

}  // namespace colmap
