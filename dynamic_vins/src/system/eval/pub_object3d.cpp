/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <iostream>

#include "utils/io/io_utils.h"
#include "utils/utils.h"

namespace dynamic_vins{\

/**
 * 可视化3D目标检测的结果
 * @param argc
 * @param argv
 * @return
 */
int PubObject3D(int argc, char **argv)
{
    const string object3d_root_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/ground_truth/kitti_tracking_object/0003/";

    auto names = GetDirectoryFileNames(object3d_root_path);

    for(auto &name:names){
        string file_path = object3d_root_path+name.string();
        std::ifstream fp(file_path);
        string line;
        int index=0;
        while (getline(fp,line)){ //循环读取每行数据
            vector<string> tokens;
            split(line,tokens," ");
            if(tokens[0]=="DontCare"){
                continue;
            }

            Box3D::Ptr box = std::make_shared<Box3D>(tokens);

            boxes.push_back(box);

            index++;
        }
        fp.close();



    }




}



}

int main(int argc, char **argv)
{
    if(argc != 2){
        std::cerr<<"please input: rosrun vins vins_node [cfg file]"<< std::endl;
        return 1;
    }

    return dynamic_vins::PubObject3D(argc,argv);
}



