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

#include "utils/box3d.h"
#include "front_end/semantic_image.h"

namespace dynamic_vins{\


class Detector3D{
public:
    using Ptr=std::shared_ptr<Detector3D>;
    explicit Detector3D(const std::string& config_path);

    void Launch(SemanticImage &img);

    std::vector<Box3D::Ptr> WaitResult();

    static std::vector<Box3D::Ptr> ReadBox3dFromTxt(const std::string &txt_path,double score_threshold);

    static std::vector<Box3D::Ptr> ReadBox3D(unsigned int seq_id);
private:
    int image_seq_id{};

};



}

#endif //DYNAMIC_VINS_DETECTOR3D_H
