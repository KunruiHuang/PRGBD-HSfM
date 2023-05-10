// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "util/math.h"

namespace colmap {

size_t NChooseK(const size_t n, const size_t k) {
  if (k == 0) {
    return 1;
  }

  return (n * NChooseK(n - 1, k - 1)) / k;
}

float InterpolateSubPixel(const float* data, const float x, const float y,
                          const int w, const int h) {
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

float InterpolateSubPixel(const cv::Mat& data, const Eigen::Vector2f& pt) {
  const float* ptr_data = reinterpret_cast<const float*>(data.data);
  return InterpolateSubPixel(ptr_data, pt[0], pt[1], data.cols, data.rows);
}

void ComputeImageGradient(const cv::Mat& data, cv::Mat& grad_out,
                          int direction) {
  // compute gradient manually using finite differences
  // direction: 0=x-direction, 1=y-direction
  const int w = data.cols;
  const int h = data.rows;
  const float* ptr_data = reinterpret_cast<const float*>(data.data);
  grad_out.setTo(0);
  float* ptr_grad_out = reinterpret_cast<float*>(grad_out.data);

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

float CalculateMean(const cv::Mat& data) {
  const float* ptr_data = reinterpret_cast<const float*>(data.data);
  float avg = 0.0f;
  for (size_t i = 0; i < data.total(); ++i) avg += ptr_data[i];
  return avg / static_cast<float>(data.total());
}

float CalculateStdDev(const cv::Mat& data, const float mean) {
  const float* ptr_data = reinterpret_cast<const float*>(data.data);
  float variance = 0.0f;
  for (size_t i = 0; i < data.total(); ++i)
    variance += (ptr_data[i] - mean) * (ptr_data[i] - mean);
  variance = variance / static_cast<float>(data.total());
  return std::sqrt(variance);
}

float CalculateStdDev(const cv::Mat& data) {
  const float mean = CalculateMean(data);
  return CalculateStdDev(data, mean);
}

float CalculateHuberWeight(const float residual, const float huber_k) {
  return std::abs(residual) <= huber_k ? 1.0f : huber_k / std::abs(residual);
}

}  // namespace colmap
