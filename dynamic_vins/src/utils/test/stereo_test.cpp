/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "stereo_test.h"

#include "utils/def.h"
#include "utils/camera_model.h"
#include "utils/io/visualization.h"


namespace dynamic_vins{\



void StereoTest(const SemanticImage &img){
    PointCloud pc;
    int rows = img.color0.rows;
    int cols = img.color0.cols;
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            int mask_value = img.merge_mask.at<uchar>(i,j);
            if(mask_value<0.5){
                continue;
            }

            float disparity = img.disp.at<float>(i,j);
            if(disparity<=0)
                continue;
            float depth = cam_s.fx0 * cam_s.baseline / disparity;//根据视差计算深度
            float x_3d = (j- cam_s.cx0)*depth/cam_s.fx0;
            float y_3d = (i-cam_s.cy0)*depth/cam_s.fy0;
            auto pixel = img.color0.at<cv::Vec3b>(i,j);
            PointT p(pixel[2],pixel[1],pixel[0]);
            p.x = x_3d;
            p.y = y_3d;
            p.z = depth;
            pc.points.push_back(p);
        }
    }

    cout<<"stereo_point_cloud:"<<pc.size()<<endl;

    PointCloudPublisher::Pub(pc,"/dynamic_vins/stereo_point_cloud");
}




}
