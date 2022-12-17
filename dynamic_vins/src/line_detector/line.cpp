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


void FrameLines::UndistortedLineEndPoints(camodocal::CameraPtr &cam)
{
    un_lines = lines;

    for (unsigned int i = 0; i < lines.size(); i++){

        Vec2d a(lines[i].StartPt.x, lines[i].StartPt.y);
        Vec3d b;
        cam->liftProjective(a, b);//将特征点反投影到归一化平面，并去畸变
        un_lines[i].StartPt.x = b.x();
        un_lines[i].StartPt.y = b.y();

        a<<lines[i].EndPt.x,lines[i].EndPt.y;
        cam->liftProjective(a, b);//将特征点反投影到归一化平面，并去畸变
        un_lines[i].EndPt.x = b.x();
        un_lines[i].EndPt.y = b.y();
    }
}

}

