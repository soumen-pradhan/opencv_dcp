#include "dcp.h"
#include <opencv2/imgproc.hpp>

auto gray(cv::Mat bgr) -> cv::Mat
{
    cv::Mat grayed(bgr.cols, bgr.rows, bgr.type());
    cv::cvtColor(bgr, grayed, cv::COLOR_BGR2GRAY);
    return grayed;
}

auto dark_channel(cv::Mat bgr, i32 size) -> cv::Mat
{
     cv::Mat channels[3]; // b, g, r
     cv::split(bgr, channels);

     size = std::max(size, 1);
     cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, { size, size });

     cv::Mat dark;
     cv::min(channels[2], channels[1], dark); // dark <- min(r, g)
     cv::min(channels[0], dark, dark); // dark <- min(b, dark)
     cv::erode(dark, dark, kernel);

     return dark;
}

auto airlight(cv::Mat bgr, cv::Mat dark) -> cv::Scalar
{
    return cv::Scalar();
}
