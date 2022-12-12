#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <vector>
#include <iostream>

#include "dcp.h"

using MatPair = std::pair<cv::Mat, cv::Mat>;

const std::string win_name = "Rail Station";

auto main() -> i32
{
    cv::Mat img = cv::imread("D:/dev/cpp/opencv_dcp/images/Rail_Station.jpg");
    cv::Mat dehaze(img.cols, img.rows, img.type());

    cv::namedWindow(win_name, cv::WINDOW_NORMAL);

    const i32 scale = 1;
    cv::resizeWindow(win_name, { img.cols / scale, img.rows / scale });

    MatPair ip_op { img, dehaze };
    auto on_trackbar = [](i32 pos, void* data) {
        auto& [img, dehaze] = *static_cast<MatPair*>(data);
        dehaze = dark_channel(img, pos);
        cv::imshow(win_name, dehaze);
    };

    cv::createTrackbar(
        "Erode Size", win_name,
        nullptr, 120,
        on_trackbar, &ip_op);

    cv::waitKey();
    cv::destroyAllWindows();
}