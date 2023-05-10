#include "rgbd/dataset_rgbd.h"

#include <fstream>
#include <iomanip>
#include <iostream>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "rgbd/camera_model.h"
#include "util/misc.h"
#include "util/pfm.h"

namespace colmap {

DatasetRGBD::DatasetRGBD() {}

DatasetRGBD::DatasetRGBD(const CameraModel& cam) : cam_(cam) {}

const CameraModel& DatasetRGBD::camera() const {
  return cam_;
}

std::shared_ptr<Frame> DatasetRGBD::loadFrame(const std::string& image_path,
                                              const std::string& depth_path) {
  cv::Mat gray_prev = loadGray(image_path);
  cv::Mat depth_prev = loadDepth(depth_path);

  std::shared_ptr<Frame> f =
      std::make_shared<Frame>(cam_.width(), cam_.height());
  f->fill(gray_prev, depth_prev);
  return f;
}

cv::Mat DatasetRGBD::loadGray(const std::string& image_path,
                              const double resolution) {
  cv::Mat gray8 = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  cv::Mat gray;
  gray8.convertTo(gray, CV_32FC1, resolution);
  return gray;
}

cv::Mat DatasetRGBD::loadDepth(const std::string& depth_path) const {
  cv::Mat read_depth;
  if (HasFileExtension(depth_path, ".pfm")) {
    read_depth = LoadPFM(depth_path);
  } else {
    cv::Mat depth16 =
        cv::imread(depth_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    depth16.convertTo(read_depth, CV_32FC1, (1.0 / 1000.0));
  }

  cv::Mat depth;
  cv::resize(read_depth, depth, cv::Size(cam_.width(), cam_.height()), cv::INTER_LINEAR);

//  cv::Mat depth16 =
//      cv::imread(depth_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
//  cv::Mat raw_depth;
//  depth16.convertTo(raw_depth, CV_32FC1, (1.0 / 1000.0));
//  cv::Mat depth;
//  cv::resize(raw_depth, depth, cv::Size(cam_.width(), cam_.height()),
//             cv::INTER_LINEAR);
  return depth;
}

/**
 * @brief   Theshold a depth image, i.e. set all values smaller
 *          than depth_min and greater than depth_max to zero.
 * @param[in,out]   depth       Depth map to be thresholded.
 * @param[in]       depth_min   Minimum depth value.
 * @param[in]       depth_max   Maximum depth value.
 */
void DatasetRGBD::threshold(cv::Mat& depth, float depth_min,
                            float depth_max) const {}

}  // namespace colmap