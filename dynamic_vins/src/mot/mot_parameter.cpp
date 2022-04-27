//
// Created by chen on 2022/4/25.
//

#include "mot_parameter.h"

#include <opencv2/opencv.hpp>

namespace dynamic_vins{\

void MotParameter::SetParameters(const std::string &config_path)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:" + config_path));
    }
    std::string kBasicDir;
    fs["basic_dir"] >> kBasicDir;

    fs["extractor_model_path"] >> kExtractorModelPath;
    kExtractorModelPath = kBasicDir + kExtractorModelPath;
    fs["tracking_n_init"] >> kTrackingNInit;
    fs["tracking_max_age"] >> kTrackingMaxAge;


    fs.release();

}


}
