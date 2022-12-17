/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "convert_utils.h"


namespace dynamic_vins{\



cv::Scalar BgrColor(const string &color_str,bool is_norm){
    cv::Scalar color;
    color[3]=1.;
    if(color_str=="white"){
        color[0]=1.;
        color[1]=1.;
        color[2]=1.;
    }
    else if(color_str=="black"){
        color[0]=0;
        color[1]=0;
        color[2]=0;
    }
    else if(color_str=="gray"){
        color[0]=0.5;
        color[1]=0.5;
        color[2]=0.5;
    }
    else if(color_str=="blue"){
        color[0]=1.;
        color[1]=0;
        color[2]=0;
    }
    else if(color_str=="green"){
        color[0]=0;
        color[1]=1.;
        color[2]=0;
    }
    else if(color_str=="red"){
        color[0]=0;
        color[1]=0;
        color[2]=1.;
    }
    else if(color_str=="yellow"){//红绿混合
        color[0]= 0;
        color[1]= 1;
        color[2]= 1;
    }
    else if(color_str=="cyan"){//青色,蓝绿混合
        color[0]= 1;
        color[1]= 1;
        color[2]= 0;
    }
    else if(color_str=="magenta"){//品红,红蓝混合
        color[0]= 1;
        color[1]= 0;
        color[2]= 1;
    }
    else{
        color[0]=1.;
        color[1]=1.;
        color[2]=1.;
    }

    if(!is_norm){
        color[0] = color[0] * 255;
        color[1] = color[1] * 255;
        color[2] = color[2] * 255;
    }

    return color;
}



}


