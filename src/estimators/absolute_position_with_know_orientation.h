#ifndef COLMAP_ABSOLUTE_POSITION_WITH_KNOW_ORIENTATION_H
#define COLMAP_ABSOLUTE_POSITION_WITH_KNOW_ORIENTATION_H

#include <array>
#include <vector>

#include <Eigen/Core>

#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

class AbsolutePositionWithKnowOrientationEstimator {
 public:
  // 归一化相机坐标
  typedef Eigen::Vector2d X_t;
  // 经过旋转后的世界坐标系下面的3D点
  typedef Eigen::Vector3d Y_t;
  // 求解得到的模型world_to_camera position
  typedef Eigen::Vector3d M_t;

  static const int kMinNumSamples = 2;

  static std::vector<M_t> Estimate(const std::vector<X_t>& points2D,
                                   const std::vector<Y_t>& points3D);

  static void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& proj_matrix, std::vector<double>* residuals);
};

}  // namespace colmap

#endif  // COLMAP_ABSOLUTE_POSITION_WITH_KNOW_ORIENTATION_H
