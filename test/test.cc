#include <iostream>
#include <string>
#include <flann/flann.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/rotation.h>
#include <ceres/ceres.h>

#include <glog/logging.h>

#include "util/misc.h"
#include "util/csv.h"
#include "util/math.h"
#include "base/pose.h"
#include "base/camera.h"
#include "rgbd/rgbd.h"
#include "base/cost_functions.h"
#include "estimators/absolute_position_with_know_orientation.h"

#include "test_global_rotation.h"

using namespace colmap;

int main(int argc, char** argv) {




  return 0;
}