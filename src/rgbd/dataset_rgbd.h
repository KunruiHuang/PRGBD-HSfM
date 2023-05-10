#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#include "rgbd/camera_model.h"
#include "rgbd/frame.h"

namespace colmap {

class DatasetRGBD {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DatasetRGBD();

  DatasetRGBD(const CameraModel& cam);

  const CameraModel& camera() const;

  std::shared_ptr<Frame> loadFrame(const std::string& image_path,
                                   const std::string& depth_path);

  cv::Mat loadGray(const std::string& image_path,
                   const double resolution = 1.0 / 1000.0);

  cv::Mat loadDepth(const std::string& depth_path) const;

  /**
   * @brief   Theshold a depth image, i.e. set all values smaller
   *          than depth_min and greater than depth_max to zero.
   * @param[in,out]   depth       Depth map to be thresholded.
   * @param[in]       depth_min   Minimum depth value.
   * @param[in]       depth_max   Maximum depth value.
   */
  void threshold(cv::Mat& depth, float depth_min, float depth_max) const;

 private:
  CameraModel cam_;
};

}  // namespace colmap
