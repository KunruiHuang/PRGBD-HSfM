#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace colmap {

float interpolate(const float* data, const float x, const float y, const int w,
                  const int h);

float interpolate(const cv::Mat& data, const Eigen::Vector2f& pt);

void computeGradient(const cv::Mat& data, cv::Mat& grad_out,
                     const int direction);

float calculateMean(const cv::Mat& data);

float calculateStdDev(const cv::Mat& data, const float mean);

float calculateStdDev(const cv::Mat& data);

float calculateHuberWeight(const float residual, const float huber_k);

}  // namespace colmap
