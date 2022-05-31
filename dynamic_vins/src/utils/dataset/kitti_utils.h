/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_KITTI_UTILS_H
#define DYNAMIC_VINS_KITTI_UTILS_H

#include "utils/parameters.h"

namespace dynamic_vins::kitti{\

    inline static std::vector<std::string> KittiLabel = {
        "Car", "Van", "Truck","Pedestrian", "Person_sitting", "Cyclist","Tram", "Misc"};

    static std::string GetKittiName(int class_id){
        if(class_id==-1){
            return "DontCare";
        }
        else{
            return KittiLabel[class_id];
        }
    }

    static int GetKittiLabelIndex(const string &name){
        for(int i=0;i<KittiLabel.size();++i){
            if(KittiLabel[i]==name){
                return i;
            }
        }
        return -1;
    }


    std::map<std::string,Eigen::MatrixXd> ReadCalibFile(const std::string &path);
}

#endif //DYNAMIC_VINS_KITTI_UTILS_H
