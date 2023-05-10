#pragma once

#include <iostream>
#include <string>

#include <Eigen/Dense>

namespace colmap {

// 估计两张RGBD图像对之间的相对Pose
bool EstimateRelativePoseFromRGBD(
    const std::string& prev_image_path,
    const std::string& prev_depth_path,
    const std::string& cur_image_path,
    const std::string& cur_depth_path,
    const double fx, const double fy,
    const double cx, const double cy,
    Eigen::Matrix3d* rotation_prev_to_cur,
    Eigen::Vector3d* translation_prev_to_cur
    );

} // namespace colmap