#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <iostream>
#include <opencv2/highgui.hpp>

#include "dcp.h"
#include <optional>

const std::string dev_path = "D:/dev/cpp/opencv_dcp/images";

auto mat_type(int type)
{
    std::string r;

    u8 depth = type & CV_MAT_DEPTH_MASK;
    u8 chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

inline auto fmtNp(cv::Mat& m) { return cv::format(m, cv::Formatter::FMT_NUMPY); }

auto showPhoto(const std::string& file)
{
    cv::Mat img = cv::imread(file);
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    std::string win_name = file;
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    const i32 scale = 1;
    cv::resizeWindow(win_name, { img.cols / scale, img.rows / scale });

    cv::Mat dark, estimate, recover, refined;
    cv::Scalar airlight;

    {
        Timer timer([](auto d) { fmt::print("Dark: {}ms\n", d.count()); });
        dark = dark_channel(img, 1);
        //cv::imshow("dark", dark);
    }
    {
        Timer timer([](auto d) { fmt::print("Airlight: {}ms\n", d.count()); });
        auto lightPatch = bright_patch(img, dark);
        airlight = cv::mean(img(lightPatch));
        //cv::rectangle(img, lightPatch, cv::Scalar(255, 255, 255));
        //cv::imshow("img", img);
    }
    {
        Timer timer([](auto d) { fmt::print("Estimate: {}ms\n", d.count()); });
        estimate = transmission_estimate(img, airlight);
        //cv::imshow("estimate", estimate);
    }
    // 32F or 64F
    {
        Timer timer([](auto d) { fmt::print("Refined: {}ms\n", d.count()); });
        refined = transmission_refine(img, estimate);
    }
    {
        Timer timer([](auto d) { fmt::print("Recover: {}ms\n", d.count()); });
        recover = recover_image(img, refined, airlight, 0.1f);
        cv::imshow(win_name, recover);
    }

    cv::waitKey();
    cv::destroyAllWindows();
}

auto playVideo(std::function<void(cv::Mat&)> transformer, std::optional<std::string> file = {})
{
    cv::Mat frame;
    cv::VideoCapture capture;

    //--- INITIALIZE VIDEOCAPTURE
    if (file) {
        capture.open(*file);
    } else {
        i32 deviceId = 0; // 0 = default camera
        capture.open(deviceId);
    }

    // check if we succeeded
    if (!capture.isOpened()) {
        fmt::print(stderr, "ERROR! Unable to open camera\n");
        return;
    }

    //--- GRAB AND WRITE LOOP
    fmt::print("Press any key to terminate");

    //auto savefile = fmt::format("{}/{}_Clear.mp4", dev_path, name);
    //auto codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
    //auto fps = cap.get(cv::CAP_PROP_FPS);
    //i32 frameWt = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    //i32 frameHt = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    //cv::VideoWriter out(savefile, codec, fps, { frameWt, frameHt });
    //out << frame;

    for (;;) {
        capture >> frame; //cap.read(frame);

        // check if we succeeded
        if (frame.empty()) {
            fmt::print(stderr, "ERROR! blank frame grabbed\n");
            break;
        }

        frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);
        transformer(frame);

        imshow("Live", frame);

        if (cv::waitKey(1) >= 0)
            break;
    }
}

auto main() -> i32
{
    //auto file = fmt::format("{}/Rail_Station.jpg", dev_path);
    //showPhoto(file);

    /***************************************************************************************/

    //auto file = fmt::format("{}/talcher/Road_1.mp4", dev_path);

    //playVideo([](auto& bgr) {
    //    auto dark = dark_channel(bgr, 1);
    //    auto brightPatch = bright_patch(bgr, dark);
    //    auto airlight = cv::mean(bgr(brightPatch));
    //    auto estimate = transmission_estimate(bgr, airlight);
    //    auto refined = transmission_refine(bgr, estimate);
    //    bgr = recover_image(bgr, refined, airlight, 0.1f);
    //}, file);
}
