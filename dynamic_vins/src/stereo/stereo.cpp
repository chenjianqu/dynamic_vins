/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/


#include "stereo.h"

#include "stereo_parameters.h"
#include "utils/log_utils.h"



namespace dynamic_vins{ \

MyStereoMatcher::MyStereoMatcher(const std::string &config_path,const std::string &seq_name) {
    stereo_para::SetParameters(config_path,seq_name);
}



void MyStereoMatcher::Launch(int seq) {
    img_seq_id=seq;
}


cv::Mat MyStereoMatcher::WaitResult() {

    std::string seq_str = PadNumber(img_seq_id,6);

    string path=stereo_para::kStereoPreprocessPath+stereo_para::kDatasetSequence+"/"+seq_str+".png";

    Debugs("stereo_path:{}",path);

    cv::Mat disp_raw = cv::imread(path,-1);
    cv::Mat disp;
    disp_raw.convertTo(disp, CV_32F,1./256.);
    return disp;
}

cv::Mat MyStereoMatcher::StereoMatch(){




}






}

