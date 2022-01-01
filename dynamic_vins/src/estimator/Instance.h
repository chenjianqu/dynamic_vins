//
// Created by chen on 2021/12/21.
//

#ifndef DYNAMIC_VINS_INSTANCE_H
#define DYNAMIC_VINS_INSTANCE_H


#include <unordered_map>
#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "dynamic.h"

class Estimator;


class Instance{
public:
    using Ptr=std::shared_ptr<Instance>;
    Instance()=default;
    Instance(const unsigned int frame_id_,const unsigned int &id_,Estimator* estimator):id(id_),e(estimator){
    }

    int slideWindowOld();
    int slideWindowNew();

    void initialPose();
    void setCurrentPoint3d();

    void setOptimizationParameters();
    void getOptimizationParameters();
    void setWindowPose();
    void outlierRejection();

    double reprojectionTwoFrameError(FeaturePoint &feat_j,FeaturePoint &feat_i,double depth,bool isStereo);


    void getBoxVertex(EigenContainer<Eigen::Vector3d> &vertex);


    vector<Eigen::Vector3d> point3d_curr;

    std::list<LandmarkPoint> landmarks;

    unsigned int id{0};

    bool isInitial{false};//是否已经初始化位姿了
    bool isTracking{true};//是否在滑动窗口中
    bool opt_vel{false};

    //物体的位姿
    State state[(WINDOW_SIZE + 1)]{};

    //物体的速度
    Vel3d vel,last_vel;

    Eigen::Vector3d box;
    cv::Scalar color;

    //优化过程中的变量
    double para_State[WINDOW_SIZE + 1][SIZE_POSE]{};
    double para_Speed[1][SIZE_SPEED]{};
    double para_Box[1][SIZE_BOX]{};
    double para_InvDepth[INSTANCE_FEATURE_SIZE][SIZE_FEATURE]{};//逆深度参数数组

    Estimator* e{nullptr};

};



#endif //DYNAMIC_VINS_DYNAMICFEATURE_H
