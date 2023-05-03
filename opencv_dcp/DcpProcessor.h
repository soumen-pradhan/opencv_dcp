#pragma once
#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utils.h"
#include <iostream>


auto mat_type(int type) -> std::string
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

class DcpProcessor {
public:
    DcpProcessor()
        : width(0)
        , height(0)
        , n_frame(0)
        , channels({ cv::Mat(0, 0, CV_32FC1), cv::Mat(0, 0, CV_32FC1), cv::Mat(0, 0, CV_32FC1) })
        , airlight({ 0, 0, 0 })
    {
    }

    DcpProcessor(i32 w, i32 h)
        : width(w)
        , height(h)
        , src(w, h, CV_32FC3)
        , gray_src(w, h, CV_32FC1)
        , channels({ cv::Mat(0, 0, CV_32FC1), cv::Mat(0, 0, CV_32FC1), cv::Mat(0, 0, CV_32FC1) })
        , dark(w, h, CV_32FC1)
        , darkIntg(w + 1, h + 1, CV_32FC1)
        , airlight({ 0, 0, 0 })
        , src_copy(w, h, CV_32FC3)
        , bottom_map(w, h, CV_32FC1)
        , top_map(w, h, CV_32FC1)
        , guideMean(w, h, CV_32FC1)
        , guide2Mean(w, h, CV_32FC1)
        , guideVar(w, h, CV_32FC1)
        , estmMean(w, h, CV_32FC1)
        , guideEstmMean(w, h, CV_32FC1)
        , guideEstmCovar(w, h, CV_32FC1)
        , guided_a(w, h, CV_32FC1)
        , guided_b(w, h, CV_32FC1)
        , processed(w, h, CV_32FC3)
    {
        // init bottom_map
        f32 width_sq = static_cast<f32>(w * w);

        for (i32 row = 0; row < h; row++) {
            f32* ptr = bottom_map.ptr<f32>(row);
            f32 r_sq = static_cast<f32>(row * row);

            for (i32 col = 0; col < w; col++) {
                ptr[col] = (r_sq) / width_sq;
            }
        }

        // init top_map
        cv::rotate(bottom_map, top_map, cv::ROTATE_180);
    }

    auto process(cv::Mat& img, cv::Mat& dst)
    {
        src = img.clone();
        cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);

        dark_channel(src, 1);
        calc_airlight();
        transmission_estimate(15);
        transmission_refine();
        recover_image(0.01f);

        dst = processed;
        n_frame = (n_frame + 1) % WRAP;
    }

private:
    i32 width, height;
    cv::Mat src, gray_src; // bgr

    const static u32 WRAP = 30;
    u32 n_frame = 0;

    std::array<cv::Mat, 3> channels;
    cv::Mat dark;
    auto dark_channel(cv::Mat& img, i32 erode_size) -> void
    {
        cv::split(img, channels); // b, g, r

        cv::min(channels[2], channels[1], dark); // dst <- min(b, g)
        cv::min(channels[0], dark, dark); // dst <- min(r, dst)

        if (erode_size > 1) {
            auto kernel = cv::getStructuringElement(cv::MORPH_RECT, { erode_size, erode_size });
            cv::erode(dark, dark, kernel, cv::Point(0, 0));
        }
    }

    cv::Mat darkIntg;
    auto blockSum(i32 x, i32 y, i32 w, i32 h) -> f32
    {
        i32 x1 = x + w, y1 = y + h;
        return darkIntg.at<f32>(y, x)
            - darkIntg.at<f32>(y, x1)
            + darkIntg.at<f32>(y1, x1)
            - darkIntg.at<f32>(y1, x);
    }

    cv::Scalar airlight;
    auto calc_airlight() -> void
    {
        if (n_frame != 0) {
            return;
        }

        i32 H = dark.rows, W = dark.cols;
        cv::integral(dark, darkIntg, CV_32F);
        auto patch = quadtree_subdivision({ 0, 0, W, H }, 4);
        airlight = cv::mean(src(patch));
    }

    /**
     *   +-------+-------+
     *   |       |       |
     *   |   A   |   B   |
     *   |       |       |
     *   +-------A------A+B
     *   |       |       |
     *   |   D   |   C   |
     *   |       |       |
     *   +------A+D---A+B+C+D
     */
    auto quadtree_subdivision(cv::Rect patch, u32 depth) -> cv::Rect
    {
        if (depth <= 0 || patch.width <= 1 || patch.height <= 1) {
            return patch;
        }

        static const std::array<std::pair<i32, i32>, 4> idx { { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } } };

        auto [x0, y0, w, h] = patch;
        i32 w2 = (w + 1) / 2, h2 = (h + 1) / 2;

        i32 xs[2] = { x0, x0 + w / 2 };
        i32 ys[2] = { y0, y0 + h / 2 };

        f32 sumMax = 0;
        i32 xMax = x0, yMax = y0;

        for (const auto& [a, b] : idx) {
            auto sum = blockSum(xs[a], ys[b], w2, h2);
            if (sum > sumMax) {
                sumMax = sum;
                xMax = xs[a];
                yMax = ys[b];
            }
        }

        auto lPatch = cv::Rect(xMax, yMax, w2, h2);
        //cv::rectangle(src, lPatch, cv::Scalar(0, 255, 0), 2);

        return quadtree_subdivision(lPatch, depth - 1);
    }

    cv::Mat src_copy, bottom_map, top_map;
    auto transmission_estimate(i32 halo_size) -> void
    {
        auto [ar, ag, ab, _] = airlight.val;
        cv::Scalar inv_airlight { 1 / ar, 1 / ag, 1 / ab };
        cv::multiply(src, inv_airlight, src_copy); // 8ms

        // 1 - omega * Dark(est)
        const f32 omega = 0.95f;
        dark_channel(src_copy, halo_size); // 5ms

        cv::multiply(dark, -1 * cv::Scalar(omega), dark); // 2ms
        //cv::multiply(dark, bottom_map, dark, -1, CV_32FC1);

        cv::add(dark, cv::Scalar::all(1), dark); // 0ms
    }

    auto transmission_refine() -> void
    {
        const i32 radius = 60;
        const f32 epsilon = 1e-4f;
        guided_filter(radius, epsilon);
    }

    cv::Mat guideMean, guide2Mean, guideVar;
    cv::Mat estmMean;
    cv::Mat guideEstmMean, guideEstmCovar;
    cv::Mat guided_a, guided_b;
    auto guided_filter(i32 radius, f32 epsilon) -> void
    {
        cv::Size kSize(radius, radius);
        cv::Mat guide = gray_src, estimate = dark;

        cv::blur(guide, guideMean, kSize);
        cv::blur(estimate, estmMean, kSize);
        cv::blur(guide.mul(estimate), guideEstmMean, kSize);

        guideEstmCovar = guideEstmMean - guideMean.mul(estmMean);
        cv::blur(guide.mul(guide), guide2Mean, kSize);
        guideVar = guide2Mean - guideMean.mul(guideMean);

        guided_a = guideEstmCovar / (guideVar + cv::Scalar::all(epsilon));
        guided_b = estmMean - guided_a.mul(guideMean);

        cv::blur(guided_a, guided_a, kSize);
        cv::blur(guided_b, guided_b, kSize);

        dark = guided_a.mul(guide) + guided_b;
    }

    cv::Mat processed;
    auto recover_image(f32 minMapVal) -> void
    {
        dark = cv::max(dark, cv::Scalar::all(minMapVal));

        // A + (I - A) / t
        cv::subtract(src, airlight, processed);

        cv::split(processed, channels);
        {
            cv::divide(channels[0], dark, channels[0]);
            cv::divide(channels[1], dark, channels[1]);
            cv::divide(channels[2], dark, channels[2]);
        }
        cv::merge(channels.data(), 3, processed);

        cv::add(processed, airlight, processed);
    }
};

/*
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
*/