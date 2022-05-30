/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_NUSCENES_H
#define DYNAMIC_VINS_NUSCENES_H

#include <string>
#include "kitti_utils.h"

namespace dynamic_vins{\


class NuScenes {
public:
    inline static vector<std::string> NuScenesLabel={"car", "truck", "trailer", "bus", "construction_vehicle",
                                                     "bicycle", "motorcycle", "pedestrian", "traffic_cone",
                                                     "barrier"};

    inline static std::map<std::string,std::set<std::string>> NuScenesToKitti = {
            {"car",{"Car","Van","Truck"}},
            {"truck",{"Car","Van","Truck"}},
            {"trailer",{"Car","Van","Truck"}},
            {"bus",{"Car","Van","Truck"}},
            {"construction_vehicle",{"Car","Van","Truck"}},
            {"bicycle",{"Cyclist"}},
            {"motorcycle",{"Cyclist"}},
            {"pedestrian",{"Pedestrian","Person_sitting"}},
            {"traffic_cone",{"traffic_cone"}},
            {"barrier",{"barrier"}}
    };

    static int ConvertNuScenesToKitti(int nuscenes_id){
        std::string nuscenes_string = NuScenesLabel[nuscenes_id];
        std::string kitti_string = * NuScenesToKitti[nuscenes_string].begin();
        return kitti::GetKittiLabelIndex(kitti_string);
    }

    static std::string GetClassName(int class_id){
        return NuScenesLabel[class_id];
    }

};


}

#endif //DYNAMIC_VINS_NUSCENES_H
