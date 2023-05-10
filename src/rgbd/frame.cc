#include "rgbd/frame.h"

#include "rgbd/math.h"

namespace colmap {

Frame::Frame(int width, int height) :
                                      width_(width),
                                      height_(height),
                                      gray_(height, width, CV_32FC1),
                                      depth_(height, width, CV_32FC1),
                                      time_color_(0.0),
                                      time_depth_(0.0),
                                      gradient_x_(height, width, CV_32FC1),
                                      gradient_y_(height, width, CV_32FC1) {}

Frame::Frame(const Frame&& other) :
                                    width_(other.width_),
                                    height_(other.height_),
                                    gray_(other.gray_),
                                    depth_(other.depth_),
                                    time_color_(other.time_color_),
                                    time_depth_(other.time_depth_),
                                    gradient_x_(other.gradient_x_),
                                    gradient_y_(other.gradient_y_) {}

Frame::~Frame() = default;

int Frame::width() const {
  return width_;
}

int Frame::height() const {
  return height_;
}

const cv::Mat& Frame::gray() const {
  return gray_;
}

cv::Mat& Frame::gray() {
  return gray_;
}

const cv::Mat& Frame::depth() const {
  return depth_;
}

cv::Mat& Frame::depth() {
  return depth_;
}

double Frame::timeColor() const {
  return time_color_;
}

void Frame::setTimeColor(double t) {
  time_color_ = t;
}

double Frame::timeDepth() const {
  return time_depth_;
}

void Frame::setTimeDepth(double t) {
  time_depth_ = t;
}

const cv::Mat& Frame::gradientX() const {
  return gradient_x_;
}

const cv::Mat& Frame::gradientY() const {
  return gradient_y_;
}

void Frame::fill(const cv::Mat& gray, const cv::Mat& depth, double time_color,
          double time_depth) {
  if (gray.empty() || depth.empty() ||
      gray.cols != width_ || gray.rows != height_ ||
      depth.cols != width_ || depth.rows != height_)
    return;

  // fill internal data from input
  size_t byte_size = static_cast<size_t>(width_) * static_cast<size_t>(height_) * sizeof(float);
  memcpy(gray_.data, gray.data, byte_size);
  memcpy(depth_.data, depth.data, byte_size);
  time_color_ = time_color;
  time_depth_ = time_depth;
  // compute gradient of intensity
  computeGradients();
}

void Frame::computeGradients() {
  computeGradient(gray_, gradient_x_, 0);
  computeGradient(gray_, gradient_y_, 1);
}

} // namespace colmap