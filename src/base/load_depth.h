#pragma once 

#include <opencv2/core/core.hpp>
#include <string>

namespace colmap {

cv::Mat LoadDepth(const std::string& depth_path, const int width,
                  const int height); 

} // namespace colmap

