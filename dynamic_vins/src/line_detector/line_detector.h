/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_LINE_DETECTOR_H
#define DYNAMIC_VINS_LINE_DETECTOR_H

#include <string>
#include "line_descriptor_custom.hpp"
#include "line.h"

namespace dynamic_vins{\

using cv::line_descriptor::KeyLine;

class LineDetector {
public:
    using Ptr=std::shared_ptr<LineDetector>;

    explicit LineDetector(const std::string &config_file);

    FrameLines::Ptr Detect(cv::Mat &img,const cv::Mat &mask=cv::Mat());

    void TrackLeftLine(const FrameLines::Ptr& prev_lines, const FrameLines::Ptr& curr_lines);

    void TrackRightLine(const FrameLines::Ptr& left_lines, const FrameLines::Ptr& right_lines);

    void TrackLine(vector<unsigned int> &curr_line_ids,
                                 vector<cv::line_descriptor::KeyLine> &curr_keylsd,
                                 cv::Mat &curr_lbd,
                                 std::map<unsigned int,int> &curr_track_cnt,
                                 vector<unsigned int> &prev_line_ids,
                                 vector<cv::line_descriptor::KeyLine> &prev_keylsd,
                                 cv::Mat &prev_lbd,
                                 std::map<unsigned int,int> &prev_track_cnt);

    static void VisualizeLineMatch(cv::Mat imageMat1, cv::Mat imageMat2,
                              std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                              std::vector<bool> good_matches);

    static void VisualizeLine(cv::Mat &img,const FrameLines::Ptr &lines);

    static void VisualizeRightLine(cv::Mat &img,const FrameLines::Ptr &lines,bool vertical);

    static void NearbyLineTracking(const std::vector<Line> &forw_lines, const std::vector<Line> &cur_lines,
                                          std::vector<std::pair<int, int> > &lineMatches) ;

private:
    cv::Ptr<cv::line_descriptor::LSDDetectorC> lsd_detector;
    cv::line_descriptor::LSDDetectorC::LSDOptions lsd_opts;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd_descriptor;
    cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> line_matcher;


    unsigned int global_line_id=0;
};


}

#endif //DYNAMIC_VINS_LINE_DETECTOR_H
