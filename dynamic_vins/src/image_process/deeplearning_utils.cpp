//
// Created by chen on 2022/10/28.
//

#include "deeplearning_utils.h"

#include "utils/parameters.h"

namespace dynamic_vins{\

camodocal::CameraPtr left_cam_dl,right_cam_dl;



void InitDeepLearningUtils(const string& config_path){
    vector<string> cam_paths = GetCameraPath(config_path);
    if(cam_paths.empty()){
        cerr<<"FeatureTracker() GetCameraPath() not found camera config:"<<config_path<<endl;
        std::terminate();
    }

    left_cam_dl = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam_paths[0]);
    if(cfg::is_stereo){
        if(cam_paths.size()==1){
            cerr<<"FeatureTracker() GetCameraPath() not found right camera config:"<<config_path<<endl;
            std::terminate();
        }
        right_cam_dl = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam_paths[1]);
    }


}



}
