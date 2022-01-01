//
// Created by chen on 2021/10/8.
//

#ifndef DYNAMIC_VINS_INSTANCE_TRACKER_H
#define DYNAMIC_VINS_INSTANCE_TRACKER_H



#include <queue>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <thread>
#include <random>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "SegmentImage.h"
#include "../parameters.h"
#include "../estimator/dynamic.h"

#include "../InstanceTracking/DeepSORT.h"

using Slice = torch::indexing::Slice;

class InstFeat{
public:
    using Ptr=std::shared_ptr<InstFeat>;
    InstFeat();
    InstFeat(unsigned int id_, int class_id_);

    unsigned int id{0};
    int class_id{0};
    cv::Scalar color;

    vector<cv::Point2f> last_points;
    std::map<unsigned int, cv::Point2f> prev_id_pts;

    vector<cv::Point2f> curr_points;
    vector<cv::Point2f> curr_un_points;
    vector<int> track_cnt;//每个特征点的跟踪次数

    std::map<unsigned int, cv::Point2f> curr_id_pts;

    vector<unsigned int> ids;

    vector<cv::Point2f> right_points;
    vector<cv::Point2f> right_un_points;
    std::map<unsigned int, cv::Point2f> right_prev_id_pts;
    std::map<unsigned int, cv::Point2f> right_curr_id_pts;

    vector<unsigned int> right_ids;

    vector<cv::Point2f> pts_velocity, right_pts_velocity;

    std::list<std::pair<cv::Point2f,cv::Point2f>> visual_points_pair;
    std::list<std::pair<cv::Point2f,cv::Point2f>> visual_right_points_pair;
    std::list<cv::Point2f> visual_new_points;

    cv::Point2f feats_center_pt;//当前跟踪的特征点的中心坐标
    int row{},col{};

    cv::Point2f box_min_pt,box_max_pt,box_center_pt;//边界框的两个点
    cv::Point2f box_vel;
    double last_time{-1.},delta_time{};

    int lost_num{0};//无法被跟踪的帧数,超过一定数量该实例将被删除

    cv::Mat mask_img;//物体在当前帧的mask
    cv::cuda::GpuMat mask_img_gpu;//物体在当前帧的mask
    torch::Tensor mask_tensor;
    float mask_area{0.};//当前帧中属于该物体的像素数量

    unsigned int last_frame_cnt{0};

};






class InstsFeatManager {
public:
    using Ptr=std::shared_ptr<InstsFeatManager>;
    InstsFeatManager();

    void instsTrack(SegImage img);
    InstancesFeatureMap setOutputFeature();
    void addViodeInstances(SegImage &img);
    cv::Mat addInstances(SegImage &img);
    void addInstancesGPU(const SegImage &img);
    void addInstancesByTracking( SegImage &img);

    void visualizeInst(cv::Mat &img);
    void drawInsts(cv::Mat& img);


    std::unordered_map<unsigned int,InstFeat> instances;
    std::unordered_map<unsigned int,Vel3d> vel_map;

    camodocal::CameraPtr camera;
    camodocal::CameraPtr right_camera;

    bool isStereo{false};
    unsigned int global_frame_id{0};
    cv::Mat mask_background;
    cv::cuda::GpuMat mask_background_gpu;

    std::vector<cv::Point2f> visual_new_points;
    SegImage prev_img;
    bool isHaveInst{false};

private:
    void manageInstances();
    vector<uchar> rejectWithF(InstFeat &inst, int col, int row) const;

    static void ptsVelocity(double dt,vector<unsigned int> &ids,vector<cv::Point2f> &curr_un_pts,std::map<unsigned int, cv::Point2f> &prev_id_pts,
                            std::map<unsigned int, cv::Point2f> &output_cur_id_pts,vector<cv::Point2f> &output_velocity);
    static float getMaskIoU(const torch::Tensor &mask1,const InstInfo &instInfo1,const float mask1_area,
                            const torch::Tensor &mask2,const InstInfo &instInfo2,const float mask2_area);
    std::tuple<int,float,float> getMatchInst(InstInfo &instInfo,torch::Tensor &inst_mask_tensor);

    unsigned long global_id_count{0};//全局特征序号，注意与静态物体上的特征id不共用
    unsigned int global_instance_id{0};

    double curr_time{},last_time{};

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lkOpticalFlow;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lkOpticalFlowBack;
    cv::Ptr<cv::cuda::CornersDetector> detector;

    DeepSORT::Ptr tracker;
};



#endif //DYNAMIC_VINS_INSTANCE_TRACKER_H
