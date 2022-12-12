#pragma once

/* Fixed Width Types */
#include <cstdint>

using i8 = int8_t;
using u8 = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;

#include <opencv2/core.hpp>

auto gray(cv::Mat bgr) -> cv::Mat;

auto dark_channel(cv::Mat bgr, i32 size = 15) -> cv::Mat;
auto airlight(cv::Mat bgr, cv::Mat dark) -> cv::Scalar;