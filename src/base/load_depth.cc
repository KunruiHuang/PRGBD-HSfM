#include "base/load_depth.h"

#include "util/logging.h"
#include "util/misc.h"
#include "util/pfm.h"

#include <opencv2/opencv.hpp>

namespace colmap {
    
cv::Mat LoadDepth(const std::string& depth_path, const int width,
                  const int height) {
   CHECK_GT(width, 0); 
   CHECK_GT(height, 0); 
   if (!ExistsFile(depth_path)) {
     throw std::runtime_error("Depth path doesn't exists."); 
   }                 
   cv::Mat read_depth;
   if (HasFileExtension(depth_path, ".pfm")) {
     read_depth = LoadPFM(depth_path);
   } else {
     cv::Mat depth16 =
         cv::imread(depth_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
     depth16.convertTo(read_depth, CV_32FC1, (1.0 / 1000.0));
   }

  cv::Mat depth;
  cv::resize(read_depth, depth, cv::Size(width, height), cv::INTER_LINEAR);

  return depth;
}

} // namespace colmap
