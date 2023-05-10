#pragma once

#include <Eigen/Core>
#include <vector>
#include <string>

namespace colmap {

struct ArkitPose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double time_stamp;
  std::string image_name;
  std::string depth_name;
  int height;
  int width;
  int tof_height;
  int tof_width;
  Eigen::Vector4d rotation;
  Eigen::Vector3d position;
  std::vector<double> intrinsics;
  std::vector<double> distortions;
};

struct Pose {
  Eigen::Vector4d qvec = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector3d tvec = Eigen::Vector3d::Zero();

  Pose Inverse() const;
};


} // namespace colmap
