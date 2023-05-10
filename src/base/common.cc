
#include "base/common.h"
#include "base/pose.h"

namespace colmap {

Pose Pose::Inverse() const {
  Pose inv_pose;
  InvertPose(qvec, tvec, &inv_pose.qvec, &inv_pose.tvec);
  return inv_pose;
}

} // namespace colmap
