#ifndef COLMAP_PFM_H
#define COLMAP_PFM_H

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>

namespace colmap {

cv::Mat LoadPFM(const std::string& file_path);

bool SavePFM(const cv::Mat image, const std::string& file_path);

} // namespace colmap

#endif  // COLMAP_PFM_H
