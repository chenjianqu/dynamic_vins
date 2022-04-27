//
// Created by chen on 2022/4/25.
//

#ifndef DYNAMIC_VINS_CAMERA_MODEL_H
#define DYNAMIC_VINS_CAMERA_MODEL_H

#include <memory>
#include <string>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

namespace dynamic_vins{\


class PinHoleCamera{
public:
    using Ptr=std::shared_ptr<PinHoleCamera>;
    PinHoleCamera(){}


    bool readFromYamlFile(const std::string& filename);
    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;

    float fx,fy,cx,cy;
    float baseline;
    int image_width,image_height;
    float k1,k2,p1,p2;//畸变矫正
    float inv_k11,inv_k22,inv_k13,inv_k23;//用于反投影
    std::string camera_name;
};


inline std::shared_ptr<PinHoleCamera> cam0;
inline std::shared_ptr<PinHoleCamera> cam1;


void InitCamera(const std::string& config_path);

}

#endif //DYNAMIC_VINS_CAMERA_MODEL_H
