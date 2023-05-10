#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "rgbd/camera_model.h"
#include "rgbd/frame.h"
#include "rgbd/pyramid.h"

namespace colmap {

class Tracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief   Tracker config struct, which is used to
   *          configure a tracker in the constructor.
   */
  struct Config {
   public:
    int num_levels = 5;
    int min_level = 1;
    int max_level = 4;
    int num_iterations = 20;
    float update_thres = 0.00001f;
    bool use_huber_weights = true;
    bool use_depth_weighting = true;

    /// print out the configuration values
    void print() const;

    /**
     * @brief   Validate the Tracker configuration,
     *          i.e. check whether the configuration is valid and
     *          correct it (or reset it) in case it is invalid.
     */
    void validate();
  };

  /**
   * @brief   Tracker constructor.
   *          Creates a tracker using a Tracker configuration and
   *          the camera model parameters.
   */
  Tracker(const Config& cfg, const CameraModel& cam);

  /// Destructor.
  ~Tracker();

  /// Get the internal tracker configuration.
  const Config& config() const;

  /**
   * @brief   Aligns two input RGB-D frame pyramids.
   * @param[in] prev_pyramid
   *          Frame pyramid of previous frame.
   * @param[in] cur_pyramid
   *          Frame pyramid of current frame.
   * @param[in,out] pose_prev_to_cur
   *          Estimated relative pose that transforms from the
   *          previous frame to the current frame.
   * @return  true if alignment successful, otherwise false.
   */
  bool align(const FramePyramid& prev_pyramid, const FramePyramid& cur_pyramid,
             Eigen::Matrix4f& pose_prev_to_cur);

 private:
  // delete copy and move constructors
  Tracker(const Tracker& other) = delete;
  Tracker(const Tracker&& other) = delete;
  // delete copy and move operators
  Tracker& operator=(const Tracker& other) = delete;
  Tracker& operator=(const Tracker&& other) = delete;

  /**
   * @brief   Calculates the per-pixel residuals using
   *          the underlying photometric cost function.
   */
  void calculateResiduals(const cv::Mat& gray_ref, const cv::Mat& depth_ref,
                          const cv::Mat& gray_cur, const CameraModel& cam,
                          const Eigen::Affine3f& pose_prev_to_cur,
                          cv::Mat& residuals) const;

  /**
   * @brief   Calculates the Jacobian with the partial derivatives
   *          for a single pixel.
   * @return  6x1 Jacobian vector for a pixel.
   */
  Eigen::Matrix<float, 6, 1> calculateJacobian(
      const float g_x, const float g_y, const CameraModel& cam,
      const Eigen::Vector3f& pt3_cur) const;

  /**
   * @brief   Calculates the weight of a residual, from the standard
   *          deviation (for Huber weight) and the depth of the pixel.
   * @return  Weight computed from Huber weight and depth-based weight.
   */
  float calculateResidualWeight(const float residual,
                                const float residuals_stddev,
                                const float depth) const;

  /**
   * @brief   Compute the update of the twist coordinates by
   *          buildung and solving the normal equations.
   * @return  6x1 update vector of the twist coordinates.
   */
  Eigen::Matrix<float, 6, 1> computeUpdate(
      const Frame& prev_frame, const Frame& cur_frame, const int level,
      const Eigen::Matrix<float, 6, 1>& xi);

  Config cfg_;
  CameraModelPyramid cam_;
  ImagePyramid residuals_;
};

}  // namespace colmap
