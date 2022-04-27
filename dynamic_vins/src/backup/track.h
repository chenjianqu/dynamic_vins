//
// Created by chen on 2021/11/30.
//

#ifndef DYNAMIC_VINS_TRACK_H
#define DYNAMIC_VINS_TRACK_H

#include <vector>
#include <cfloat>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "utils/parameters.h"

struct Track {
    int id{};
    cv::Rect2f box{};
};


namespace {


// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
inline cv::Rect2f get_rect_xysr(const cv::Mat &xysr) {
    auto cx = xysr.at<float>(0, 0), cy = xysr.at<float>(1, 0), s = xysr.at<float>(2, 0), r = xysr.at<float>(3, 0);
    float w = sqrt(s * r);
    float h = s / w;
    float x = (cx - w / 2);
    float y = (cy - h / 2);

    return {x, y, w, h};
}


void draw_text(cv::Mat &img, const std::string &str,
               const cv::Scalar &color, cv::Point pos, bool reverse = false) {
    auto t_size = cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, nullptr);
    cv::Point bottom_left, upper_right;
    if (reverse) {
        upper_right = pos;
        bottom_left = cv::Point(upper_right.x - t_size.width, upper_right.y + t_size.height);
    } else {
        bottom_left = pos;
        upper_right = cv::Point(bottom_left.x + t_size.width, bottom_left.y - t_size.height);
    }

    cv::rectangle(img, bottom_left, upper_right, color, -1);
    cv::putText(img, str, bottom_left, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255) - color);
}



void draw_bbox(cv::Mat &img, cv::Rect2f bbox,
               const std::string &label = "", const cv::Scalar &color = {0, 0, 0}) {
    cv::rectangle(img, bbox, color);
    if (!label.empty()) {
        draw_text(img, label, color, bbox.tl());
    }
}



cv::Scalar color_map(int64_t n) {
    auto bit_get = [](int64_t x, int64_t i) { return x & (1 << i); };

    int64_t r = 0, g = 0, b = 0;
    int64_t i = n;
    for (int64_t j = 7; j >= 0; --j) {
        r |= bit_get(i, 0) << j;
        g |= bit_get(i, 1) << j;
        b |= bit_get(i, 2) << j;
        i >>= 3;
    }
    return cv::Scalar(b, g, r);
}


}


#endif //DYNAMIC_VINS_TRACK_H
