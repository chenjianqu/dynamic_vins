//
// Created by chen on 2022/6/13.
//

#ifndef DYNAMIC_VINS_IMAGE_PROCESS_H
#define DYNAMIC_VINS_IMAGE_PROCESS_H

#include <memory>

#include "front_end/semantic_image.h"
//#include "flow/flow_estimator.h"
#include "det3d/detector3d.h"
#include "det2d/detector2d.h"
#include "stereo/stereo.h"


namespace dynamic_vins{\

class ImageProcessor{
public:
    using Ptr=std::shared_ptr<ImageProcessor>;

    explicit ImageProcessor(const std::string &config_file);

    static std::vector<Box3D::Ptr> BoxAssociate2Dto3D(std::vector<Box3D::Ptr> &boxes3d,std::vector<Box2D::Ptr> &boxes2d);


    void Run(SemanticImage &img);


    Detector2D::Ptr detector2d;
    Detector3D::Ptr detector3d;
    //FlowEstimator::Ptr flow_estimator;
    MyStereoMatcher::Ptr stereo_matcher;

};


}

#endif //DYNAMIC_VINS_IMAGE_PROCESS_H
