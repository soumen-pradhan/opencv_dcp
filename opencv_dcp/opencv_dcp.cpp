#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <iostream>
#include <optional>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "DcpProcessor.h"
#include "dcp.h"

const std::string dev_path = "D:/dev/cpp/opencv_dcp/images";

auto fmtNp(const cv::Mat& m) { return cv::format(m, cv::Formatter::FMT_NUMPY); }
auto fmtCsv(const cv::Mat& m) { return cv::format(m, cv::Formatter::FMT_CSV); }

auto weightMat(i32 width, i32 height, cv::Mat& dst)
{
    dst.create(height, width, CV_32FC1);
    f32 width_sq = static_cast<f32>(width * width);

    /*// 538 microsec
    for (i32 row = 0; row < height; row++) {
        f32 r = static_cast<f32>(row);

        for (i32 col = 0; col < width; col++) {
            mat.at<f32>(row, col) = (r * r) / width_sq;
        }
    }
    */

    // 300 microsec
    for (i32 row = 0; row < height; row++) {
        f32* ptr = dst.ptr<f32>(row);
        f32 r = static_cast<f32>(row);

        for (i32 col = 0; col < width; col++) {
            ptr[col] = (r * r) / width_sq;
        }
    }

    /* // 5000 microsec
    cv::parallel_for_({ 0, height }, [&](const cv::Range& range) {
        for (i32 row = range.start; row < range.end; row++) {
            f32* ptr = mat.ptr<f32>(row);
            f32 r = static_cast<f32>(row);

            for (i32 col = 0; col < width; col++) {
                ptr[col] = (r * r) / width_sq;
            }
        }
    });
    */
}

auto showPhoto(const std::string& file)
{
    cv::Mat img = cv::imread(file);
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    std::string win_name = file;
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    const i32 scale = 1;
    cv::resizeWindow(win_name, { img.cols / scale, img.rows / scale });

    auto loc = fmt::format("{}/Rail_Station_stages/", dev_path);

    cv::Mat dark;
    {
        Timer<> timer([](auto d) { fmt::print("Dark: {}ms\n", d.count()); });
        dark_channel(img, dark, 15);
        //cv::imwrite(loc + "01-Dark.Channel_size15.exr", dark);
    }
    cv::Scalar airlight;
    cv::Rect lightPatch;
    cv::Mat img_clone = img.clone();
    {
        Timer<> timer([](auto d) { fmt::print("Airlight: {}ms\n", d.count()); });
        lightPatch = bright_patch(img_clone, dark);
        airlight = cv::mean(img(lightPatch));
        img_clone.convertTo(img_clone, CV_8UC3, 255, 0);
        cv::imwrite(loc + "02-airlight_rec4.jpg", img_clone);
    }
    {
        Timer<> timer([](auto d) { fmt::print("Estimate: {}ms\n", d.count()); });
        transmission_estimate(img, airlight, dark);
        //cv::imwrite(loc + "03-estimate_omega0.95.exr", dark);
    }
    cv::Mat guide; // for sharp edges but 1 channel
    cv::cvtColor(img, guide, cv::COLOR_BGR2GRAY);
    {
        Timer<> timer([](auto d) { fmt::print("Refined: {}ms\n", d.count()); });
        transmission_refine(guide, dark, dark);
        //cv::imwrite(loc + "04-refine_radius60_epsilon1e-4.exr", dark);
    }
    {
        Timer<> timer([](auto d) { fmt::print("Recover: {}ms\n", d.count()); });
        recover_image(img, dark, airlight, img);
        //cv::imwrite(loc + "05-recover_minVal0.1.exr", img);
    }

    //std::cout << "lightpatch: " << lightPatch << "\n";
    //cv::rectangle(img, lightPatch, cv::Scalar(0, 0, 255), 5);

    img.convertTo(img, CV_8UC3, 255, 0);
    cv::imshow(win_name, img);

    cv::waitKey();
    cv::destroyAllWindows();
}

auto playVideo(std::optional<std::string> file, std::function<void(cv::Mat&)> transformer)
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

    /** How to save stream
        auto savefile = fmt::format("{}/{}_Clear.mp4", dev_path, name);
        auto codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
        auto fps = cap.get(cv::CAP_PROP_FPS);
        i32 frameWt = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        i32 frameHt = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        cv::VideoWriter out(savefile, codec, fps, { frameWt, frameHt });
        out << frame;    
    */

    for (u64 i = 1;; i++) {
        capture >> frame; // cap.read(frame);

        // check if we succeeded
        if (frame.empty()) {
            fmt::print(stderr, "ERROR! blank frame grabbed\n");
            break;
        }

        frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);
        {
            Timer<> time([&](auto d) { fmt::print("\r{:3} ms", d.count()); });
            transformer(frame);
        }
        frame.convertTo(frame, CV_8UC1, 255, 0);

        imshow("Live", frame);
        //auto out = fmt::format("{}/talcher/Road_4_frames/{}.jpg", dev_path, i);
        //cv::imwrite(out, frame);

        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
}

auto playVideoWithProcessor(std::optional<std::string> file)
{
    cv::Mat frame, dehazed;
    cv::VideoCapture capture;

    // INITIALIZE VIDEOCAPTURE
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
    fmt::print("Press any key to terminate\n");

    /*  How to save stream
        auto savefile = fmt::format("{}/{}_Clear.mp4", dev_path, name);
        auto codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
        auto fps = cap.get(cv::CAP_PROP_FPS);
        i32 frameWt = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        i32 frameHt = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        cv::VideoWriter out(savefile, codec, fps, { frameWt, frameHt });
        out << frame;    
    */

    i32 frameWt = static_cast<i32>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    i32 frameHt = static_cast<i32>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Rect skyView(0, 0, frameWt, frameHt / 3);

    DcpProcessor dcp_proc(frameWt, frameHt);

    cv::Mat bottomMap, topMap;
    weightMat(frameWt, frameHt, bottomMap);
    cv::rotate(bottomMap, topMap, cv::ROTATE_180);

    for (u64 i = 1;; i++) {
        capture >> frame; // cap.read(frame);

        // check if we succeeded
        if (frame.empty()) {
            fmt::print(stderr, "ERROR! blank frame grabbed\n");
            break;
        }

        u64 tot = 0;
        frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);
        {
            //Timer<> time([&](auto d) { fmt::print("\r{:3} ms", d.count()); });
            dcp_proc.process(frame, dehazed);
            cv::blendLinear(dehazed, frame, topMap, bottomMap, dehazed);

            // TODO smoothen sky
        }
        dehazed.convertTo(dehazed, CV_8UC1, 255, 0);

        cv::imshow("Live", dehazed);
        //auto out = fmt::format("{}/talcher/Road_4_frames/{}.jpg", dev_path, i);
        //cv::imwrite(out, frame);

        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
}

auto mean_linear(cv::Mat& src, cv::Mat& dst, i32 kSize)
{
    CV_Assert(kSize % 2 == 1); // blur kernel size should be odd

    dst.create(src.cols, src.rows, src.type());

    // padding is 0
    cv::Mat intg;
    cv::integral(src, intg, CV_32F);

    for (i32 row = 0; row < src.rows; row++) {
        i32 top_row = std::max(row - (kSize / 2), 0);
        i32 bottom_row = std::min(top_row + kSize, src.rows);

        f32* top_ptr = intg.ptr<f32>(top_row);
        f32* bottom_ptr = intg.ptr<f32>(bottom_row);
        f32* dst_ptr = dst.ptr<f32>(row);

        i32 left_x = -(kSize / 2);
        i32 right_x = left_x + kSize;

        for (i32 col = 0; col < src.cols; col++, left_x++, right_x++) {
            f32 top_left = top_ptr[std::max(left_x, 0)];
            f32 top_right = top_ptr[std::min(right_x, src.cols)];
            f32 bottom_left = bottom_ptr[std::max(left_x, 0)];
            f32 bottom_right = bottom_ptr[std::min(right_x, src.cols)];

            dst_ptr[col] = (top_left - top_right - bottom_left + bottom_right);
        }
    }
}

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include <immintrin.h>

void mean_linear_simd_chatgpt(cv::Mat& src, cv::Mat& dst, int kSize)
{
    CV_Assert(kSize % 2 == 1); // blur kernel size should be odd

    dst.create(src.size(), src.type());

    const int simd_size = 8; // AVX2 has 8 floats per vector

    // padding is 0
    cv::Mat intg;
    cv::integral(src, intg, CV_32F);

    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    for (int row = 0; row < src.rows; row++) {
        int top_row = std::max(row - (kSize / 2), 0);
        int bottom_row = std::min(top_row + kSize, src.rows);

        float* top_ptr = intg.ptr<float>(top_row);
        float* bottom_ptr = intg.ptr<float>(bottom_row);
        float* dst_ptr = dst.ptr<float>(row);

        int left_x = -(kSize / 2);
        int right_x = left_x + kSize;

        for (int col = 0; col < src.cols; col += simd_size, left_x += simd_size, right_x += simd_size) {
            __m256 top_left = _mm256_i32gather_ps(top_ptr + std::max(left_x, 0), indices, sizeof(float));
            __m256 top_right = _mm256_i32gather_ps(top_ptr + std::min(right_x, src.cols), indices, sizeof(float));
            __m256 bottom_left = _mm256_i32gather_ps(bottom_ptr + std::max(left_x, 0), indices, sizeof(float));
            __m256 bottom_right = _mm256_i32gather_ps(bottom_ptr + std::min(right_x, src.cols), indices, sizeof(float));

            __m256 sum = _mm256_sub_ps(top_left, top_right);
            sum = _mm256_sub_ps(sum, bottom_left);
            sum = _mm256_add_ps(sum, bottom_right);

            __m256 div = _mm256_set1_ps(1.0f / (kSize * kSize));
            __m256 res = _mm256_mul_ps(sum, div);

            _mm256_storeu_ps(dst_ptr + col, res);
        }

        // Handle remaining columns
        for (int col = src.cols - (src.cols % simd_size); col < src.cols; col++) {
            float top_left = top_ptr[std::max(left_x, 0)];
            float top_right = top_ptr[std::min(right_x, src.cols)];
            float bottom_left = bottom_ptr[std::max(left_x, 0)];
            float bottom_right = bottom_ptr[std::min(right_x, src.cols)];

            dst_ptr[col] = (top_left - top_right - bottom_left + bottom_right) / (kSize * kSize);

            left_x++;
            right_x++;
        }
    }
}

auto mean_linear_simd(cv::Mat& src, cv::Mat& dst, i32 kSize) -> void
{
    CV_Assert(kSize % 2 == 1); // odd kernel size
    dst.create(src.cols, src.rows, src.type());
    dst.setTo(cv::Scalar::all(0));

    const i32 simd_size = 8;
    f32 kSizeDiv = 1.0f / static_cast<f32>(kSize * kSize);

    cv::Mat intg;
    cv::integral(src, intg, CV_32F);
    //cv::multiply(intg, cv::Scalar(kSizeDiv), intg);

    // only middle portion of frame
    for (i32 row = kSize / 2; row < src.rows - (kSize / 2); row++) {
        i32 top_row = row - (kSize / 2);
        i32 bottom_row = top_row + kSize;

        f32* top_ptr = intg.ptr<f32>(top_row);
        f32* bottom_ptr = intg.ptr<f32>(bottom_row);
        f32* dst_ptr = dst.ptr<f32>(row);

        i32 left_x = -(kSize / 2);
        i32 right_x = left_x + kSize;

        for (i32 col = kSize / 2; col < src.cols - (kSize / 2); col += simd_size, left_x += simd_size, right_x += simd_size) {
            __m256 reg1 = _mm256_loadu_ps(top_ptr + left_x); // top_left
            __m256 reg2 = _mm256_loadu_ps(bottom_ptr + right_x); // bottom_right

            reg1 = _mm256_add_ps(reg1, reg2);

            reg2 = _mm256_loadu_ps(top_ptr + right_x); // top_right
            reg1 = _mm256_sub_ps(reg1, reg2);

            reg2 = _mm256_loadu_ps(bottom_ptr + left_x); // bottom_left
            reg1 = _mm256_sub_ps(reg1, reg2);

            //_mm256_div_ps(reg1, ) // divide

            _mm256_storeu_ps(dst_ptr + col, reg1);
        }

        // remaing cols
    }

    // remaining rows
}

auto blur_cmp(const std::string& file)
{

    cv::Mat img = cv::imread(file);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    img.convertTo(img, CV_32FC1, 1.0 / 255.0);

    i32 kSz = 21;
    cv::Size kSize(kSz, kSz);
    cv::Rect view(20, 20, 10, 10);

    //{
    //    mean_linear(img, mean_lin, kSize);
    //    mean_lin.convertTo(mean_lin, CV_8UC3, 255, 0);
    //    cv::imshow("linear", mean_lin);
    //}
    /***************************************************************************************/
    cv::Mat box;
    {
        Timer<> timer([&](auto d) { fmt::print("Box: {}ms\n", d.count()); });
        cv::boxFilter(img, box, CV_32F, kSize);
    }
    box.convertTo(box, CV_8UC1, 255, 0);
    cv::imshow("Box", box);
    /***************************************************************************************/
    cv::Mat blur;
    {
        Timer<> timer([&](auto d) { fmt::print("Blur: {}ms\n", d.count()); });
        cv::blur(img, blur, kSize);
    }
    blur.convertTo(blur, CV_8UC1, 255, 0);
    cv::imshow("Blur", blur);
    /***************************************************************************************/
    //cv::Mat median;
    //{
    //    Timer<> timer([&](auto d) { fmt::print("Median: {}ms\n", d.count()); });
    //    cv::medianBlur(img, median, 21);
    //}
    //median.convertTo(median, CV_8UC1, 255, 0);
    //cv::imshow("Blur", median);
    /***************************************************************************************/
    cv::Mat gauss;
    {
        Timer<> timer([&](auto d) { fmt::print("Gauss: {}ms\n", d.count()); });
        cv::GaussianBlur(img, gauss, kSize, 0);
    }
    gauss.convertTo(gauss, CV_8UC1, 255, 0);
    cv::imshow("Gauss", gauss);
    /***************************************************************************************/
    img.convertTo(img, CV_8UC1, 255, 0);
    cv::imshow("img", img);

    cv::waitKey();
    cv::destroyAllWindows();
}

auto PSNR()
{
    std::string fyp_img = "C:/Users/SOUMEN/Desktop/dehaze_img/fyp_images";

    f64 tot = 0;

    for (u32 i = 1; i <= 18; i++) {
        fmt::print("done\n");
        auto og = fmt::format("{}/SOTS_outdoor/0{}.jpg", fyp_img, i);
        auto og_img = cv::imread(og);

        auto res = fmt::format("{}/SOTS_outdoor_results/0{}.jpg", fyp_img, i);
        auto res_img = cv::imread(res);

        auto score = cv::PSNR(og_img, res_img);
        tot += score;
    }

    //fmt::print("Indoor {}\n", tot / 50);
    //Indoor 9.604886810711758

    //fmt::print("Outdoor {}\n", tot / 50);
    //Outdoor 4.2023153687071835

    cv::waitKey();
    cv::destroyAllWindows();
}

auto main() -> i32
{
    /*
    img = cv::imread(file);
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);
    fmt::print("img {}x{}\n", img.rows, img.cols);
    DcpProcessor dcp_proc(img.rows, img.cols);
    dcp_proc.process(img, dehazed);


    auto createWindow = [&](const std::string& name, i32 scale = 1) {
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::resizeWindow(name, { img.cols / scale, img.rows / scale });
    };

    createWindow(file, 4);


    //cv::imshow(file, dehazed);

    //auto map = weightMat({ img.cols, img.rows },
    //    [=](i32 r, i32 c) { return std::pow(static_cast<f32>(r) / img.rows, 2.5f); });
    //cv::Mat inv_map;
    //cv::rotate(map, inv_map, cv::ROTATE_180);
    //cv::Rect view(100, 100, 5, 5);
    //std::cout << fmtNp(map(view));
    //cv::imshow(file, map);

    //cv::imshow(file, blend);

    auto callback = [](i32 pos, void* data) {
        fmt::print("\r {} %", pos);
        auto bottom = weightMat({ img.cols, img.rows },
            [=](i32 r, i32 c) { return std::pow(static_cast<f32>(r) / img.rows, pos / 1000.f); });

        cv::Mat top;
        cv::rotate(bottom, top, cv::ROTATE_180);

        cv::blendLinear(dehazed, img, top, bottom, blend);
        cv::imshow(file, blend);
    };

    cv::createTrackbar("Og", file, nullptr, 2000, callback);
    cv::setTrackbarPos("Og", file, trackbarPos);


    cv::waitKey();
    cv::destroyAllWindows();
    */

    //showPhoto(file);

    cv::Mat ip, blur, blur2, blur3;
    ip = cv::Mat::eye(5, 5, CV_32FC1);
    std::cout << fmtNp(ip) << "\n\n";

    cv::blur(ip, blur, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_CONSTANT);
    std::cout << fmtNp(blur) << "\n\n";

    mean_linear(ip, blur2, 3);
    std::cout << fmtNp(blur2) << "\n\n";

    mean_linear_simd(ip, blur3, 3);
    std::cout << fmtNp(blur3) << "\n\n";

    u64 cv_tot = 0, mean_tot = 0;

    //for (u32 i = 0; i < 100; i++) {
    //    cv::randu(ip, { 0 }, { 1 });
    //    {
    //        Timer<> timer([&](auto micro) { cv_tot += micro.count(); });
    //        cv::blur(ip, blur, cv::Size(61, 61), cv::Point(-1, -1), cv::BORDER_CONSTANT);
    //    }
    //    {
    //        Timer<> timer([&](auto micro) { mean_tot += micro.count(); });
    //        mean_linear(ip, blur2, 61);
    //    }
    //}
    //fmt::print("  cv: {:5} micro\nmean: {:5} micro", cv_tot / 100.0f, mean_tot / 100.0f);

    /*
    const i32 ARRAY_SIZE = 8;

    // Declare arrays
    f32 a[ARRAY_SIZE] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    f32 b[ARRAY_SIZE] = { 0 };
    f32 c[ARRAY_SIZE] = { 0 };

    __m256 a_avx = _mm256_loadu_ps(a);
    __m256 b_avx = _mm256_loadu_ps(b);
    __m256 c_avx = _mm256_mul_ps(a_avx, a_avx);
    _mm256_storeu_ps(c, c_avx);

    fmt::print("c: {}\n", fmt::ptr(c));
    fmt::print("c + 3: {}\n", fmt::ptr(c + 3));
    fmt::print("&c[3]: {}\n", fmt::ptr(&c[3]));
    */

    /***************************************************************************************/

    //auto file = fmt::format("{}/nit/20230130_062620.mp4", dev_path);
    //playVideoWithProcessor(file);

    /*
    playVideo(file, [](auto& img) {
        cv::Mat dark;
        dark_channel(img, dark, 1);

        auto brightPatch = bright_patch(img, dark);
        auto airlight = cv::mean(img(brightPatch));

        // halo around objects due to size param in estimate
        transmission_estimate(img, airlight, dark);

        cv::Mat guide;
        cv::cvtColor(img, guide, cv::COLOR_BGR2GRAY);
        transmission_refine(guide, dark, dark);

        recover_image(img, dark, airlight, img);

        //cv::rectangle(img, brightPatch, cv::Scalar(0, 0, 255), 1);
    });
    */
}

// __m128 -> 4 f32
// __m256 -> 8 f32