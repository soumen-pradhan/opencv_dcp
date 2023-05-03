#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <opencv2/imgproc.hpp>
#include <utility>

#include "dcp.h"

auto quadtree_subdivision( const cv::Mat& src,
    const cv::Mat& integral, cv::Rect patch, u32 depth) -> cv::Rect;

auto guided_filter(
    const cv::Mat& guide, const cv::Mat& estimate, cv::Mat& dst, i32 radius, f32 epsilon) -> void;

auto dark_channel(const cv::Mat& src, cv::Mat& dst, i32 size) -> void
{
    cv::Mat channels[3]; // r, g, b
    cv::split(src, channels);

    cv::min(channels[2], channels[1], dst); // dst <- min(b, g)
    cv::min(channels[0], dst, dst); // dst <- min(r, dst)

    if (size > 1) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, { size, size });
        cv::erode(dst, dst, kernel, cv::Point(0, 0));
    }
}

auto bright_patch(const cv::Mat& src, const cv::Mat& dark) -> cv::Rect
{
    i32 H = dark.rows, W = dark.cols;

    cv::Mat darkIntg;
    cv::integral(dark, darkIntg, CV_32F);

    return quadtree_subdivision(src, darkIntg, { 0, 0, W, H }, 4);
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
auto quadtree_subdivision(const cv::Mat& src,
    const cv::Mat& integral, cv::Rect patch, u32 depth) -> cv::Rect
{
    if (depth <= 0 || patch.width <= 1 || patch.height <= 1) {
        return patch;
    }

    const std::array<std::pair<i32, i32>, 4> idx { { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } } };

    const auto blockSum = [&](i32 x, i32 y, i32 w, i32 h) {
        i32 x1 = x + w, y1 = y + h;
        return integral.at<f32>(y, x)
            - integral.at<f32>(y, x1)
            + integral.at<f32>(y1, x1)
            - integral.at<f32>(y1, x);
    };

    auto [x0, y0, w, h] = patch;
    i32 w2 = (w + 1) / 2, h2 = (h + 1) / 2;

    i32 xs[2] = { x0, x0 + w / 2 };
    i32 ys[2] = { y0, y0 + h / 2 };

    f32 sumMax = 0;
    i32 xMax = x0, yMax = y0;

    for (auto& [a, b] : idx) {
        auto sum = blockSum(xs[a], ys[b], w2, h2);
        if (sum > sumMax) {
            sumMax = sum;
            xMax = xs[a];
            yMax = ys[b];
        }
    }

    auto lPatch = cv::Rect(xMax, yMax, w2, h2);
    cv::rectangle(src, lPatch, cv::Scalar(0, 255, 0), 2);

    return quadtree_subdivision(src, integral, lPatch, depth - 1);
}

auto transmission_estimate(
    const cv::Mat& src, const cv::Scalar& airlight, cv::Mat& dst, i32 size) -> void
{
    auto [ar, ag, ab, _] = airlight.val;
    cv::Scalar inv_airlight { 1 / ar, 1 / ag, 1 / ab };
    {
        // Timer timer([](auto d) { LOGI("RGB / A:  %lldms", d.count()); });
        // cv::divide(rgb, airlight, estimate); // 12ms
        cv::multiply(src, inv_airlight, dst); // 8ms
    }

    // 1 - omega * Dark(est)
    f32 omega = 0.95f;

    {
        // Timer timer([](auto d) { LOGI("Dark(Est): %lldms", d.count()); });
        dark_channel(dst, dst, size); // 5ms
    }

    cv::multiply(dst, -1 * cv::Scalar(omega), dst); // 2ms
    cv::add(dst, cv::Scalar(1), dst); // 0ms
}

auto transmission_refine(const cv::Mat& guide, const cv::Mat& estimate, cv::Mat& dst)
    -> void
{
    i32 radius = 60;
    f32 epsilon = 1e-4f;
    guided_filter(guide, estimate, dst, radius, epsilon);
}

auto guided_filter(
    const cv::Mat& guide, const cv::Mat& estimate, cv::Mat& dst, i32 radius, f32 epsilon) -> void
{
    cv::Size kSize(radius, radius);

    cv::Mat guideMean, estmMean, guideEstmMean;
    {
        //Timer timer([&](auto d) { fmt::print("I_mean: %{}ms", d.count()); });
        cv::boxFilter(guide, guideMean, CV_32F, kSize); // 3ms
    }
    {
        //Timer timer([&](auto d) { fmt::print("p_mean: %{}ms", d.count()); });
        cv::boxFilter(estimate, estmMean, CV_32F, kSize); // 3ms
    }
    {
        //Timer timer([&](auto d) { fmt::print("Ip_mean: %{}ms", d.count()); });
        cv::boxFilter(guide.mul(estimate), guideEstmMean, CV_32F, kSize); // 4ms
    }

    auto guideEstmCovariance = guideEstmMean - guideMean.mul(estmMean);

    cv::Mat guide2Mean;
    {
        //Timer timer([&](auto d) { fmt::print("I2_mean: %{}ms", d.count()); });
        cv::boxFilter(guide.mul(guide), guide2Mean, CV_32F, kSize); // 4ms
    }
    auto guideVariance = guide2Mean - guideMean.mul(guideMean);

    cv::Mat a = guideEstmCovariance / (guideVariance + cv::Scalar(epsilon)); // 1ms
    cv::Mat b = estmMean - a.mul(guideMean);

    {
        //Timer timer([&](auto d) { fmt::print("a_mean: %{}ms", d.count()); });
        cv::boxFilter(a, a, CV_32F, kSize); // 3ms
    }
    {
        //Timer timer([&](auto d) { fmt::print("b_mean: %{}ms", d.count()); });
        cv::boxFilter(b, b, CV_32F, kSize); // 3ms
    }

    dst = a.mul(guide) + b;
}

auto recover_image(
    const cv::Mat& src, cv::Mat& transmissionMap, const cv::Scalar& airlight,
    cv::Mat& dst, f32 minMapVal) -> void
{
    transmissionMap = cv::max(transmissionMap, cv::Scalar::all(minMapVal));

    // (I - A) / t + A

    cv::subtract(src, airlight, dst);

    {
        //cv::divide(recover, estimate, recover);
        cv::Mat channels[3];
        cv::split(dst, channels);
        {
            //Timer timer([&](auto d) { fmt::print("divide: {}ms", d.count()); });

            // cv::parallel_for_(
            //     { 0, 3 }, [&](auto rng) {
            //         for (u32 i = rng.start; i < rng.end; i++) {
            //             cv::divide(channels[i], transmissionMap, channels[i]);
            //         }
            //     }
            // );

            cv::divide(channels[0], transmissionMap, channels[0]);
            cv::divide(channels[1], transmissionMap, channels[1]);
            cv::divide(channels[2], transmissionMap, channels[2]);
        }
        cv::merge(channels, 3, dst);
    }

    cv::add(dst, airlight, dst);
}
