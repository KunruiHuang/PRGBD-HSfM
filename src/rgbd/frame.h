#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace colmap {

class Frame {
 public:
  Frame(int width, int height);

  Frame(const Frame&& other);

  ~Frame();

  // disable copy constructor
  Frame(const Frame& other) = delete;
  // disable copy assignment operator
  Frame& operator=(const Frame& other) = delete;
  // disable move assignment operator
  Frame& operator=(const Frame&& other) = delete;

  /// Get width of the RGB-D frame.
  int width() const;

  /// Get height of the RGB-D frame.
  int height() const;

  /// Return const reference to the intensity image.
  const cv::Mat& gray() const;

  /// Return reference to the intensity image.
  cv::Mat& gray();

  /// Return const reference to the depth image.
  const cv::Mat& depth() const;

  /// Return reference to the depth image.
  cv::Mat& depth();

  /// Return timestamp of the color/intensity image.
  double timeColor() const;

  /**
   * @brief Allows to set the timestamp of the color/intensity image.
   * @param t     color timestamp.
   */
  void setTimeColor(double t);

  /// Return timestamp of the depth image.
  double timeDepth() const;

  /**
   * @brief Allows to set the timestamp of the depth image.
   * @param t     depth timestamp.
   */
  void setTimeDepth(double t);

  /**
   * @return Const reference to the x-gradient of the intensity image.
   */
  const cv::Mat& gradientX() const;

  /**
   * @return Const reference to the y-gradient of the intensity image.
   */
  const cv::Mat& gradientY() const;

  /**
   * @brief Fills the RGB-D frame container with data.
   * The intensity image and depth map are not cloned but only assigned,
   * i.e. stored smart pointer in cv::Mat still points to the original images.
   * @param gray          intensity image (type CV_32FC1)
   * @param depth         depth image (type CV_32FC1)
   * @param time_color    timestamp of the color image
   * @param time_depth    timestamp of the depth image
   */
  void fill(const cv::Mat& gray, const cv::Mat& depth, double time_color = 0.0,
            double time_depth = 0.0);

  /**
   * @brief Computes both the x- and y- gradient of the intensity image
   *        using central differences.
   */
  void computeGradients();

 private:
  int width_;           ///< width of the RGB-D frame
  int height_;          ///< height of the RGB-D frame
  cv::Mat gray_;        ///< intensity image (single-channel float)
  cv::Mat depth_;       ///< depth image (single-channel float, 1.0f = 1.0m)
  double time_color_;   ///< color timestamp
  double time_depth_;   ///< depth timestamp
  cv::Mat gradient_x_;  ///< x-gradient of the intensity
  cv::Mat gradient_y_;  ///< y-gradient of the intensity
};

}  // namespace colmap
