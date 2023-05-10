#include "rgbd/rgbd.h"

#include "rgbd/camera_model.h"
#include "rgbd/dataset_rgbd.h"
#include "rgbd/frame.h"
#include "rgbd/tracker.h"

namespace colmap {

bool EstimateRelativePoseFromRGBD(const std::string& prev_image_path,
                                  const std::string& prev_depth_path,
                                  const std::string& cur_image_path,
                                  const std::string& cur_depth_path,
                                  const double fx, const double fy,
                                  const double cx, const double cy,
                                  Eigen::Matrix3d* rotation_prev_to_cur,
                                  Eigen::Vector3d* translation_prev_to_cur) {
  Eigen::Matrix3f K;
  K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  const int width = 1920;
  const int height = 1440;
  CameraModel camera(width, height, K);

  DatasetRGBD data(camera);

  Tracker::Config tracker_cfg;
  const int w = data.camera().width();
  const int h = data.camera().height();
  const int num_levels = tracker_cfg.num_levels;
  std::shared_ptr<FramePyramid> prev_pyramid =
      std::make_shared<FramePyramid>(w, h, num_levels);
  std::shared_ptr<FramePyramid> cur_pyramid =
      std::make_shared<FramePyramid>(w, h, num_levels);

  Tracker tracker(tracker_cfg, data.camera());
  std::shared_ptr<Frame> prev_frame =
      data.loadFrame(prev_image_path, prev_depth_path);
  std::shared_ptr<Frame> cur_frame =
      data.loadFrame(cur_image_path, cur_depth_path);

  prev_pyramid->fill(*prev_frame);
  cur_pyramid->fill(*cur_frame);

  Eigen::Matrix4f pose_prev_to_cur = Eigen::Matrix4f::Identity();
  bool ok = tracker.align(*prev_pyramid, *cur_pyramid, pose_prev_to_cur);

  if (!ok) {
    return false;
  }

  *rotation_prev_to_cur = pose_prev_to_cur.block<3, 3>(0, 0).cast<double>();
  *translation_prev_to_cur = pose_prev_to_cur.block<3, 1>(0, 3).cast<double>();

  return true;
}

}  // namespace colmap
