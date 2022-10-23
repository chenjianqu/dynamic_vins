/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "line.h"


#include "line_descriptor/include/line_descriptor_custom.hpp"

namespace dynamic_vins{

void FrameLines::SetLines() {
    int size = keylsd.size();
    for(int i=0;i<size;++i){
        Line l;
        l.id = line_ids[i];
        l.StartPt = keylsd[i].getStartPoint();
        l.EndPt = keylsd[i].getEndPoint();
        l.length = keylsd[i].lineLength;
        lines.push_back(l);
    }
}


void FrameLines::UndistortedLineEndPoints(PinHoleCamera::Ptr cam)
{
    un_lines = lines;

    for (unsigned int i = 0; i < lines.size(); i++){
        un_lines[i].StartPt.x = (lines[i].StartPt.x - cam->cx) / cam->fx;
        un_lines[i].StartPt.y = (lines[i].StartPt.y - cam->cy) / cam->fy;
        un_lines[i].EndPt.x = (lines[i].EndPt.x - cam->cx) / cam->fx;
        un_lines[i].EndPt.y = (lines[i].EndPt.y - cam->cy) / cam->fy;
    }
}

}

