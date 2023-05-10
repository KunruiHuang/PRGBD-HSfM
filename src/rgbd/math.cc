#include "rgbd/math.h"

namespace colmap {

float interpolate(const float* data, const float x, const float y, const int w,
                  const int h) {
  // bilinear interpolation

  // round up/down
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  // check whether point to check is in range
  if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h)
    return std::numeric_limits<float>::quiet_NaN();

  // compute differences
  float x1_diff = x - static_cast<float>(x0);
  float y1_diff = y - static_cast<float>(y0);
  float x0_diff = 1.0f - x1_diff;
  float y0_diff = 1.0f - y1_diff;
  // compute weights
  float w00 = x0_diff * y0_diff;
  float w10 = x1_diff * y0_diff;
  float w01 = x0_diff * y1_diff;
  float w11 = x1_diff * y1_diff;

  // compute average
  float avg = 0.0f;
  avg += data[y0 * w + x0] * w00;
  avg += data[y0 * w + x1] * w10;
  avg += data[y1 * w + x0] * w01;
  avg += data[y1 * w + x1] * w11;
  return avg;
}

float interpolate(const cv::Mat& data, const Eigen::Vector2f& pt) {
  const float *ptr_data = reinterpret_cast<const float *>(data.data);
  return interpolate(ptr_data, pt[0], pt[1], data.cols, data.rows);
}

void computeGradient(const cv::Mat& data, cv::Mat& grad_out,
                     const int direction) {
  // compute gradient manually using finite differences
  // direction: 0=x-direction, 1=y-direction
  const int w = data.cols;
  const int h = data.rows;
  const float *ptr_data = reinterpret_cast<const float *>(data.data);
  grad_out.setTo(0);
  float *ptr_grad_out = reinterpret_cast<float *>(grad_out.data);

  const int y_start = direction;
  const int y_end = h - y_start;
  const int x_start = 1 - direction;
  const int x_end = w - x_start;
  if (direction == 1) {
    // y-direction
    for (int y = y_start; y < y_end; ++y) {
      for (int x = x_start; x < x_end; ++x) {
        float v0 = ptr_data[(y - 1) * w + x];
        float v1 = ptr_data[(y + 1) * w + x];
        ptr_grad_out[y * w + x] = 0.5f * (v1 - v0);
      }
    }
  } else {
    // x-direction
    for (int y = y_start; y < y_end; ++y) {
      for (int x = x_start; x < x_end; ++x) {
        float v0 = ptr_data[y * w + (x - 1)];
        float v1 = ptr_data[y * w + (x + 1)];
        ptr_grad_out[y * w + x] = 0.5f * (v1 - v0);
      }
    }
  }
}

float calculateMean(const cv::Mat& data) {
  const float *ptr_data = reinterpret_cast<const float *>(data.data);
  float avg = 0.0f;
  for (size_t i = 0; i < data.total(); ++i) avg += ptr_data[i];
  return avg / static_cast<float>(data.total());
}

float calculateStdDev(const cv::Mat& data, const float mean) {
  const float *ptr_data = reinterpret_cast<const float *>(data.data);
  float variance = 0.0f;
  for (size_t i = 0; i < data.total(); ++i)
    variance += (ptr_data[i] - mean) * (ptr_data[i] - mean);
  variance = variance / static_cast<float>(data.total());
  return std::sqrt(variance);
}

float calculateStdDev(const cv::Mat& data) {
  const float mean = calculateMean(data);
  return calculateStdDev(data, mean);
}

float calculateHuberWeight(const float residual, const float huber_k) {
  return std::abs(residual) <= huber_k ? 1.0f : huber_k / std::abs(residual);
}

} // namespace colmap
