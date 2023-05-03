#pragma once

#include "utils.h"

#include <opencv2/core.hpp>

auto dark_channel(const cv::Mat& src, cv::Mat& dst, i32 size = 15) -> void;

auto bright_patch(const cv::Mat& src, const cv::Mat& dark) -> cv::Rect;

auto transmission_estimate(const cv::Mat& src, const cv::Scalar& airlight, cv::Mat& dst, i32 size = 15) -> void;

auto transmission_refine(const cv::Mat& src, const cv::Mat& estimate, cv::Mat& dst) -> void;

auto recover_image(const cv::Mat& src, cv::Mat& transmissionMap, const cv::Scalar& airlight, cv::Mat& dst, f32 minMapVal = 0.1f) -> void;