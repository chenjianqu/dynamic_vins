/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_DETECTOR3D_H
#define DYNAMIC_VINS_DETECTOR3D_H

#include <memory>

#include "basic/box3d.h"
#include "basic/semantic_image.h"

namespace dynamic_vins{\


class Detector3D{
public:
    using Ptr=std::shared_ptr<Detector3D>;
    explicit Detector3D(const std::string& config_path,const std::string& seq_name);

    void Launch(SemanticImage &img);

    std::vector<Box3D::Ptr> WaitResult();

    static std::vector<Box3D::Ptr> ReadBox3dFromTxt(const std::string &txt_path,double score_threshold);
    static std::vector<Box3D::Ptr> ReadGroundtruthFromKittiTracking(int frame);

    static std::vector<Box3D::Ptr> ReadBox3D(const string &target_name);
private:
    int image_seq_id{};
    double image_time{};

};



}

#endif //DYNAMIC_VINS_DETECTOR3D_H
