//
// Created by chen on 2021/11/19.
//
#include "SegmentImage.h"
#include "../parameters.h"
#include "../utils.h"
#include "../estimator/dynamic.h"




std::vector<uchar> flowTrack(const cv::Mat &img1,const cv::Mat &img2,vector<cv::Point2f> &pts1,vector<cv::Point2f> &pts2)
{
    std::vector<uchar> status;
    std::vector<float> err;

    if(img1.empty() || img2.empty() || pts1.empty()){
        std::string msg="flowTrack() input wrong, received at least one of parameter are empty";
        tkLogger->error(msg);
        throw std::runtime_error(msg);
    }

    cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2,status, err, cv::Size(21, 21), 3);

    //反向光流计算 判断之前光流跟踪的特征点的质量
    if(Config::FLOW_BACK){
        vector<uchar> reverse_status;
        std::vector<cv::Point2f> reverse_pts = pts1;
        cv::calcOpticalFlowPyrLK(img2, img1, pts2, reverse_pts,
                                 reverse_status, err, cv::Size(21, 21), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3);
        for(size_t i = 0; i < status.size(); i++){
            if(status[i] && reverse_status[i] && distance(pts1[i], reverse_pts[i]) <= 0.5)
                status[i] = 1;
            else
                status[i] = 0;
        }
    }

    ///将落在图像外面的特征点的状态删除
    for (size_t i = 0; i < pts2.size(); ++i){
        if (status[i] && !inBorder(pts2[i],img2.rows,img2.cols))
            status[i] = 0;
    }
    return status;
}

/**
 * 将gpu mat转换为point2f
 * @param d_mat
 * @param vec
 */
 void gpuMat2Points(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
     std::vector<cv::Point2f> points(d_mat.cols);
     cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&points[0]);
     d_mat.download(mat);
     vec = points;
}

 void gpuMat2Status(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec)
{
     std::vector<uchar> points(d_mat.cols);
     cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&points[0]);
    d_mat.download(mat);
    vec=points;
}

/**
 * 将point2f转换为gpu mat
 * @param vec
 * @param d_mat
 */
 void points2GpuMat(const std::vector<cv::Point2f>& vec, cv::cuda::GpuMat& d_mat)
{
    cv::Mat mat(1, vec.size(), CV_32FC2, (void*)&vec[0]);
    d_mat=cv::cuda::GpuMat(mat);
}

 void status2GpuMat(const std::vector<uchar>& vec,cv::cuda::GpuMat& d_mat)
{
    cv::Mat mat(1, vec.size(), CV_8UC1, (void*)&vec[0]);
    d_mat=cv::cuda::GpuMat(mat);
}


std::vector<uchar> flowTrackGpu(const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlow,
                                const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlowBack,
                                const cv::cuda::GpuMat &img_prev,const cv::cuda::GpuMat &img_next,
                                std::vector<cv::Point2f> &pts_prev,std::vector<cv::Point2f> &pts_next){
     if(img_prev.empty() || img_next.empty() || pts_prev.empty()){
         std::string msg="flowTrack() input wrong, received at least one of parameter are empty";
         tkLogger->error(msg);
         throw std::runtime_error(msg);
     }

     static auto getValidStatusSize=[](const std::vector<uchar> &stu){
         int cnt=0;
         for(const auto s : stu) if(s)cnt++;
         return cnt;
     };

    std::vector<float> err;

    cv::cuda::GpuMat d_prevPts;
    points2GpuMat(pts_prev,d_prevPts);
    cv::cuda::GpuMat d_nextPts;
    cv::cuda::GpuMat d_status;

    lkOpticalFlow->calc(img_prev,img_next,d_prevPts,d_nextPts,d_status);

    std::vector<uchar> status;
    gpuMat2Status(d_status,status);
    gpuMat2Points(d_nextPts,pts_next);

    int forward_success=getValidStatusSize(status);
    debug_t("flowTrackGpu forward success:{}",forward_success);

    //反向光流计算 判断之前光流跟踪的特征点的质量
    if(Config::FLOW_BACK){
        cv::cuda::GpuMat d_reverse_status;
        cv::cuda::GpuMat d_reverse_pts = d_prevPts;

        lkOpticalFlowBack->calc(img_next,img_prev,d_nextPts,d_reverse_pts,d_reverse_status);

        std::vector<uchar> reverse_status;
        gpuMat2Status(d_reverse_status,reverse_status);

        std::vector<cv::Point2f> pts_prev_reverse;
        gpuMat2Points(d_reverse_pts,pts_prev_reverse);

        //constexpr float SAVE_RATIO=0.2f;
        //if(int inv_success = getValidStatusSize(reverse_status); inv_success*1.0 / forward_success > SAVE_RATIO){
            for(size_t i = 0; i < reverse_status.size(); i++){
                if(status[i] && reverse_status[i] && distance(pts_prev[i], pts_prev_reverse[i]) <= 1.)
                    status[i] = 1;
                else
                    status[i] = 0;
            }
            debug_t("flowTrackGpu backward success:{}",getValidStatusSize(status));
        //}
        /*else{
            std::vector<std::tuple<int,float>> feats_dis(status.size());
            for(int i=0;i<status.size();++i){
                float d = distance(pts_prev[i], pts_prev_reverse[i]);
                feats_dis[i] = {i,d};
            }
            std::sort(feats_dis.begin(),feats_dis.end(),[](auto &a,auto &b){
                return std::get<1>(a) < std::get<1>(b);//根据dis低到高排序
            });
            const int SAVE_FEAT_NUM = forward_success * SAVE_RATIO;
            for(int i=0,cnt=0;i<status.size();++i){
                int j=std::get<0>(feats_dis[i]);
                if(status[j] && cnt<SAVE_FEAT_NUM){
                    cnt++;
                }
                else{
                    status[j]=0;
                }
            }
            tkLogger->warn("flowTrackGpu backward success:{},so save:{}",getValidStatusSize(reverse_status),getValidStatusSize(status));
        }*/

    }

    ///将落在图像外面的特征点的状态删除
    for (size_t i = 0; i < pts_next.size(); ++i){
        if (status[i] && !inBorder(pts_next[i],img_next.rows,img_next.cols))
            status[i] = 0;
    }

    debug_t("flowTrackGpu input:{} final_success:{}",status.size(),getValidStatusSize(status));

    return status;
}




std::vector<cv::Point2f> detectNewFeaturesGPU(int detect_num,const cv::cuda::GpuMat &img,const cv::cuda::GpuMat &mask)
{
    auto detector = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, detect_num, 0.01, Config::MIN_DIST);
    debug_t("start detect new feat,size:{}",detect_num);

    cv::cuda::GpuMat d_new_pts;
    detector->detect(img,d_new_pts,mask);

    debug_t("end detect new feat,size:{}",d_new_pts.cols);

    std::vector<cv::Point2f> points;
    gpuMat2Points(d_new_pts,points);

    debug_t("gpuMat2Points points:{}",points.size());

    return points;
}

/**
 * 叠加两个mask，结果写入到第一个maks中
 * @param mask1
 * @param mask2
 */
void superpositionMask(cv::Mat &mask1, const cv::Mat &mask2)
{
    for (int i = 0; i < mask1.rows; i++) {
        uchar* mask1_ptr=mask1.data+i*mask1.step;
        uchar* mask2_ptr=mask2.data+i*mask2.step;
        for (int j = 0; j < mask1.cols; j++) {
            if(mask1_ptr[0]==255 && mask2_ptr[0]<128){
                mask1_ptr[0]=0;
            }
            mask1_ptr+=1;
            mask2_ptr+=1;
        }
    }
}



void setMask(const cv::Mat &init_mask,cv::Mat &mask_out,std::vector<cv::Point2f> &points){
    mask_out = init_mask.clone();
    for(const auto& pt : points){
        cv::circle(mask_out, pt, Config::MIN_DIST, 0, -1);
    }
}


void setMaskGpu(const cv::cuda::GpuMat &init_mask,cv::cuda::GpuMat &mask_out){

}








/**
 * 对特征点进行去畸变,返回归一化特征点
 * @param pts
 * @param cam
 * @return
 */
vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (auto & pt : pts){
        Eigen::Vector2d a(pt.x, pt.y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);//将特征点反投影到归一化平面，并去畸变
        un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
    }
    return un_pts;
}



