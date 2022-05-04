/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_NUSCENES_H
#define DYNAMIC_VINS_NUSCENES_H

#include <string>

class NuScenes {
public:
    inline static vector<std::string> NuScenesLabel={"car", "truck", "trailer", "bus", "construction_vehicle",
                                                     "bicycle", "motorcycle", "pedestrian", "traffic_cone",
                                                     "barrier"};

    static std::string GetClassName(int class_id){
        return NuScenesLabel[class_id];
    }

};


#endif //DYNAMIC_VINS_NUSCENES_H
