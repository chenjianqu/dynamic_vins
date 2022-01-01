//
// Created by chen on 2021/10/8.
//

#include "instance_tracker.h"
#include "SegmentImage.h"
#include "../utils.h"
#include "../utility/ViodeUtils.h"


using namespace torch::indexing;
using std::unordered_map;
using std::make_pair;
using std::map;


std::default_random_engine randomEngine;
std::uniform_int_distribution<unsigned int> color_rd(0,255);

InstFeat::InstFeat():color(color_rd(randomEngine),color_rd(randomEngine),color_rd(randomEngine))
{}

InstFeat::InstFeat(unsigned int id_, int class_id_): id(id_), class_id(class_id_),
color(color_rd(randomEngine),color_rd(randomEngine),color_rd(randomEngine))
{}


InstsFeatManager::InstsFeatManager()
{
    lkOpticalFlow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30);
    lkOpticalFlowBack = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21),1,30,true);

    std::array<int64_t, 2> orig_dim{int64_t(Config::ROW), int64_t(Config::COL)};
    tracker = std::make_unique<DeepSORT>(orig_dim);
}


void InstsFeatManager::instsTrack(SegImage img)
{
    TicToc ticToc;
    curr_time=img.time0;



    if(Config::Dataset == DatasetType::KITTI){
        //addInstancesGPU(img);
        addInstancesByTracking(img);
        info_t("instsTrack addInstances:{} ms",ticToc.toc_then_tic());
        isHaveInst = !img.insts_info.empty();
    }
    else if(Config::Dataset == DatasetType::VIODE){
        addViodeInstances(img);
        info_t("instsTrack addViodeInstancesBySegImg:{} ms",ticToc.toc_then_tic());
    }
    else{
        string msg="Have not this Dataset Type";
        tkLogger->critical(msg);
        throw std::runtime_error(msg);
    }


    if(isHaveInst){

        ///形态学运算
        static auto erode_kernel = cv::getStructuringElement(
                cv::MORPH_RECT,cv::Size(10,10),cv::Point(-1,-1));
        static auto erode_filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE,CV_8UC1,erode_kernel);
        erode_filter->apply(img.merge_mask_gpu,img.merge_mask_gpu);
        img.merge_mask_gpu.download(img.merge_mask);


        //img.gray0_gpu.download(img.gray0);
        //img.gray1_gpu.download(img.gray1);

        for(auto& [key,inst] : instances){
            if(inst.last_frame_cnt < global_frame_id){
                inst.lost_num++;
            }
            else if(inst.last_frame_cnt == global_frame_id){
                inst.lost_num=0;
            }
            /*///将gpu图像传输到cpu，用于随机访问
            if(inst.lost_num ==0 && Config::Dataset != DatasetType::VIODE){
                inst.mask_img_gpu.download(inst.mask_img);
            }*/
        }

        info_t("instsTrack set mask_img:{} ms",ticToc.toc_then_tic());


        /*if constexpr(false){
            static cv::Mat mask_element=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(10,10),cv::Point(-1,-1));
            ticToc.tic();
            for(auto& [key,inst] : instances){
                if(inst.lost_num>0)continue;
                cv::erode(inst.mask_img, inst.mask_img, mask_element, cv::Point(-1, -1));//腐蚀，缩小物体区域
            }
            info_t("instsTrack erode:{} ms",ticToc.toc_then_tic());
        }*/

        debug_t("instsTrack prev_img.gray0_gpu:({}x{},type:{})",
                prev_img.gray0_gpu.rows,prev_img.gray0_gpu.cols,prev_img.gray0_gpu.type());
        debug_t("instsTrack img.gray0_gpu:({}x{},type:{})",
                img.gray0_gpu.rows,img.gray0_gpu.cols,img.gray0_gpu.type());

        //对每个目标进行光流跟踪
        //printf("光流跟踪: ");
        for(auto & [ key,inst] : instances){
            if(inst.last_points.empty() || inst.lost_num>0)
                continue;

            inst.curr_points.clear();

            /*///平移跟踪的图像，以减小运动物体的影响
            cv::Mat detect_img;
            int row_shift= - inst.vel.y() * inst.delta_time;
            int col_shift= - inst.vel.x() * inst.delta_time;
            printf("inst:%d vel(%.2lf,%.2lf) shift:(row:%d,col:%d)\n",pair.first,inst.vel.x(),inst.vel.y(),row_shift,col_shift);*/
            debug_t("inst:{} last_points:{} mask({}x{},type:{})",
                    inst.id,inst.last_points.size(),inst.mask_img.rows,inst.mask_img.cols,inst.mask_img.type());

            ///光流跟踪
            //auto status = flowTrack(prev_img.gray0,img.gray0,inst.last_points,inst.curr_points);
            auto status = flowTrackGpu(lkOpticalFlow,lkOpticalFlowBack,prev_img.gray0_gpu,
                                       img.gray0_gpu,inst.last_points,inst.curr_points);

            /*imageTranslate(img.gray0,detect_img,row_shift,col_shift);
            vector<uchar> status = flowTrack(last_img.gray0,detect_img,inst.last_points,inst.curr_points);
            for(auto& pt:inst.curr_points){
                pt.y -= row_shift;
                pt.x -= col_shift;
            }*/
            if(Config::Dataset == DatasetType::VIODE){
                for(size_t i=0;i<status.size();++i){
                    //if(status[i] && VIODE::pixel2key(inst.curr_points[i],img.seg0)!=key) status[i]=0;
                    if(status[i] && inst.mask_img.at<uchar>(inst.curr_points[i]) == 0) status[i]=0;
                }
            }
            else{
                for(size_t i=0;i<status.size();++i)
                    if(status[i] && inst.mask_img.at<uchar>(inst.curr_points[i]) == 0)
                        status[i]=0;
            }

            reduceVector(inst.curr_points,status);
            reduceVector(inst.ids,status);
            reduceVector(inst.last_points,status);

        }
        info_t("instsTrack flowTrack:{} ms",ticToc.toc_then_tic());


        ///对每个特征点增加观测次数
        for(auto & [ key,inst] : instances){
            if(inst.last_points.empty() || inst.lost_num>0)
                continue;
            for (auto &n : inst.track_cnt) n++;
        }

        /*
         ///RANSANC剔除点
         if constexpr (false){
            if(!instances.empty()){
                for(auto& pair: instances)
                {
                    auto &inst=pair.second;
                    if(inst.curr_points.size()<8 || inst.last_points.size()<8)
                        continue;
                    auto status=rejectWithF(inst,img.gray0.cols,img.gray0.rows);
                    reduceVector(inst.last_points, status);
                    reduceVector(inst.curr_points, status);
                    reduceVector(inst.ids, status);
                }
            }
        }
        if constexpr (false){
            for(auto & [key,inst] : instances){
                inst.visual_new_points.clear();
                if(inst.lost_num>0 || inst.curr_points.size()>INST_FEAT_NUM)
                    continue;
                vector<cv::Point2f> new_pts;
                size_t new_num=INST_FEAT_NUM - inst.curr_points.size();
                cv::goodFeaturesToTrack(img.gray0, new_pts, (int)new_num, 0.01, DYNAMIC_MIN_DIST, inst.mask_img); //检测新的角点
                for(int i=0;i<std::min(new_pts.size(),new_num);++i){
                    inst.curr_points.emplace_back(new_pts[i]);
                    inst.ids.emplace_back(global_id_count++);
                    inst.visual_new_points.emplace_back(new_pts[i]);
                }
                //printf("cur_pts:%d new_pts:%d\n",inst.curr_points.size(),new_pts.size());
            }
        }*/

        if(isHaveInst){
            ///添加新的特征点前的准备
            int max_new_detect=0;
            for(auto & [key,inst] : instances){
                if(inst.lost_num>0 || inst.curr_points.size()>= Config::MAX_DYNAMIC_CNT)
                    continue;
                max_new_detect += (Config::MAX_DYNAMIC_CNT - (int)inst.curr_points.size());
            }

            debug_t("instsTrack max_new_detect={}",max_new_detect);

            ///添加新的特征点
            if(max_new_detect > 0){
                mask_background = img.merge_mask;
                for(auto & [key,inst] : instances){
                    if(inst.lost_num>0 || inst.curr_points.size()>=Config::MAX_DYNAMIC_CNT)
                        continue;
                    inst.visual_points_pair.clear();
                    inst.visual_right_points_pair.clear();
                    inst.visual_new_points.clear();

                    for(size_t i=0;i<inst.curr_points.size();++i){
                        inst.visual_points_pair.emplace_back(inst.last_points[i],inst.curr_points[i]);//用于可视化
                        cv::circle(mask_background, inst.curr_points[i], Config::MIN_DYNAMIC_DIST, 0, -1);//设置mask
                    }
                }

                info_t("instsTrack prepare detect:{} ms",ticToc.toc_then_tic());

                /*cv::goodFeaturesToTrack(img.gray0, new_pts, (int)new_detect, 0.01, DYNAMIC_MIN_DIST, mask_bg_new); //检测新的角点
                cv::imshow("mask_background",mask_background);
                cv::waitKey(0);
                cv::cuda::threshold(img.merge_mask_gpu,img.merge_mask_gpu,0.5,255,CV_8UC1);*/
                mask_background_gpu.upload(mask_background);

                debug_t("instsTrack | mask_background size:{}x{} type:{} ",
                        mask_background.rows,mask_background.cols,mask_background.type());
                debug_t("instsTrack | mask_background_gpu size:{}x{} type:{} ",
                        mask_background_gpu.rows,mask_background_gpu.cols,mask_background_gpu.type());
                debug_t("instsTrack | img.gray0_gpu size:{}x{} type:{} ",
                        img.gray0_gpu.rows,img.gray0_gpu.cols,img.gray0_gpu.type());

                auto new_pts = detectNewFeaturesGPU(max_new_detect,img.gray0_gpu,mask_background_gpu);
                visual_new_points = new_pts;
                debug_t("instsTrack actually detect num:{}",new_pts.size());
                for(auto &pt : new_pts){
                    int index_inst=-1;
                    for(auto &[key,inst] : instances){
                        if(inst.lost_num>0 || inst.curr_points.size()>Config::MAX_DYNAMIC_CNT) continue;
                        if(inst.mask_img.at<uchar>(pt) >= 1){
                            index_inst=(int)key;
                            break;
                        }
                    }

                    if(index_inst!=-1){
                        instances[index_inst].curr_points.emplace_back(pt);
                        instances[index_inst].ids.emplace_back(global_id_count++);
                        instances[index_inst].visual_new_points.emplace_back(pt);
                        instances[index_inst].track_cnt.push_back(1);
                    }
                }
                info_t("instsTrack detectNewFeaturesGPU:{} ms",ticToc.toc_then_tic());
            }

        }


        for(auto& [key,inst] : instances){
            ///去畸变和计算归一化坐标
            inst.curr_un_points=undistortedPts(inst.curr_points, camera);
            ///计算特征点的速度
            ptsVelocity(curr_time-last_time,inst.ids,inst.curr_un_points,
                        inst.prev_id_pts,inst.curr_id_pts,inst.pts_velocity);
        }
        info_t("instsTrack undistortedPts & ptsVelocity:{} ms",ticToc.toc_then_tic());

        /// 右边相机图像的跟踪
        if((!img.gray1.empty() || !img.gray1_gpu.empty()) && isStereo){
            for(auto& [key,inst] : instances){
                inst.right_points.clear();
                if(!inst.curr_points.empty() && inst.lost_num==0){
                    //auto status= flowTrack (img.gray0,img.gray1,inst.curr_points,inst.right_points);
                    auto status = flowTrackGpu(lkOpticalFlow,lkOpticalFlowBack,img.gray0_gpu,img.gray1_gpu,
                                               inst.curr_points,inst.right_points);

                    if(Config::Dataset == DatasetType::VIODE){
                        for(size_t i=0;i<status.size();++i){
                            if(status[i] && VIODE::pixel2key(inst.right_points[i],img.seg1)!=key)
                                status[i]=0;
                        }
                    }
                    inst.right_ids = inst.ids;
                    reduceVector(inst.right_points, status);
                    reduceVector(inst.right_ids, status);
                    inst.right_un_points = undistortedPts(inst.right_points, right_camera);
                    ptsVelocity(curr_time-last_time,inst.right_ids,inst.right_un_points,
                                inst.right_prev_id_pts,inst.right_curr_id_pts,inst.right_pts_velocity);
                    //printf("Inst:%d del:%d ",inst.id,inst.right_ids.size()-inst.right_points.size());
                }
            }
        }
        info_t("instsTrack flowTrack right:{} ms",ticToc.toc_then_tic());

        manageInstances();
        info_t("instsTrack manageInstances:{} ms",ticToc.toc_then_tic());

        ///输出实例数据
        debug_t("InstanceTracker:实例数量:{}",instances.size());
        for(auto &[key,inst] : instances){
            debug_t("Inst:{} size:{}",key,inst.curr_points.size());
        }

        for(auto& [key,inst] : instances){
            inst.last_points=inst.curr_points;
            inst.prev_id_pts=inst.curr_id_pts;
            inst.right_prev_id_pts=inst.right_curr_id_pts;
        }

    }
    else
    {
        manageInstances();


        for(auto& [key,inst] : instances){
            inst.curr_points.clear();
            inst.curr_un_points.clear();
            inst.last_points.clear();
            inst.right_points.clear();
            inst.right_un_points.clear();
            inst.ids.clear();
            inst.right_ids.clear();
            inst.pts_velocity.clear();
            inst.right_pts_velocity.clear();
            inst.prev_id_pts.clear();
            inst.visual_points_pair.clear();
            inst.visual_right_points_pair.clear();
            inst.visual_new_points.clear();
            inst.track_cnt.clear();
        }
    }

    last_time=curr_time;
    global_frame_id++;

    prev_img = img;
}





/**
 *删除不被观测到的实例,设置lost_num变量是为了防止有时候深度学习算法忽略了某帧
 */
void InstsFeatManager::manageInstances()
{
    for(auto it=instances.begin(),it_next=it; it!=instances.end();it=it_next)
    {
        it_next++;
        auto &inst=it->second;

        if(inst.lost_num ==0 && inst.curr_points.empty()){
            inst.lost_num++;
        }

        if(inst.lost_num > 0){
            inst.lost_num++;
            if(inst.lost_num > 3){ //删除该实例
                instances.erase(it);
            }
        }
    }
}


/**
 ** 用于将特征点传到VIO模块
 * @param result
 */
InstancesFeatureMap InstsFeatManager::setOutputFeature()
{
    InstancesFeatureMap result;
    for(auto& [key,inst]: instances)
    {
        if(inst.lost_num>0)continue;
        InstanceFeatureSimple  featuresMap;
        featuresMap.color = inst.color;

        for(int i=0;i<(int)inst.curr_un_points.size();++i){
            Eigen::Matrix<double,5,1> feat;
            feat<<inst.curr_un_points[i].x,inst.curr_un_points[i].y, 1 ,inst.pts_velocity[i].x,inst.pts_velocity[i].y;
            vector<Eigen::Matrix<double,5,1>> vp={feat};
            featuresMap.insert({inst.ids[i],vp});
        }

        int right_cnt=0;
        if(isStereo){
            for(int i=0; i<(int)inst.right_un_points.size(); i++){
                auto r_id = inst.right_ids[i];
                if(featuresMap.count(r_id) ==0)
                    continue;
                Eigen::Matrix<double,5,1> feat;
                feat<<inst.right_un_points[i].x,inst.right_un_points[i].y, 1 ,inst.right_pts_velocity[i].x,inst.right_pts_velocity[i].y;
                featuresMap[r_id].push_back(feat);
                right_cnt++;
            }
        }

        result.insert({key,featuresMap});

        debug_t("inst_id:{} l_track:{} r_track:{}",key,featuresMap.size(),right_cnt);
    }
    return result;
}


/**
 * 根据语义标签，多线程设置mask
 * @param mask_img 原来的mask
 * @param semantic_img 语义标签图像
 */
void InstsFeatManager::addViodeInstances(SegImage &img)
{
    cv::Mat seg = img.seg0;
    debug_t("start to addViodeInstancesBySegImg()");

    for(auto &[key,inst] : instances){
        inst.mask_area=0;
    }

    debug_t("addViodeInstancesBySegImg merge insts");

    for(auto &inst_info : img.insts_info){
        auto key = inst_info.track_id;
        if(instances.count(key)==0){
            InstFeat instanceFeature(key, 0);
            instances.insert({key,instanceFeature});
        }

        auto &inst = instances[key];
        inst.mask_img = inst_info.mask_cv;
        inst.box_vel = cv::Point2f(0,0);
        inst.last_frame_cnt = global_frame_id;
        inst.last_time = img.time0;
        debug_t("inst:{} mask_img:({}x{}) local_mask:({}x{})",key,instances[key].mask_img.rows,instances[key].mask_img.cols,
                        inst.mask_img.rows,inst.mask_img.cols);
    }


    for(auto &[key,inst]: instances){
        if(inst.last_frame_cnt == global_frame_id){
            auto rect = cv::boundingRect(inst.mask_img);
            inst.box_min_pt= rect.tl();
            inst.box_max_pt = rect.br();
            inst.box_center_pt = (inst.box_min_pt+inst.box_max_pt)/2;
            inst.color = img.seg0.at<cv::Vec3b>(inst.box_center_pt);
        }
    }

    isHaveInst = !img.insts_info.empty();
}



float InstsFeatManager::getMaskIoU(const torch::Tensor &mask1,const InstInfo &instInfo1,const float mask1_area,
                                   const torch::Tensor &mask2,const InstInfo &instInfo2,const float mask2_area)
{
    auto intersection_mask=(mask1 * mask2);
    float intersection_area = intersection_mask.sum(torch::IntArrayRef({0,1})).item().toFloat();
    return intersection_area/(mask1_area + mask2_area - intersection_area);
}




/**
 *
 * @param instInfo
 * @param inst_mask_tensor
 * @param inst_mask_area
 * @return
 */
std::tuple<int,float,float> InstsFeatManager::getMatchInst(InstInfo &instInfo,torch::Tensor &inst_mask_tensor)
{
    int h=(int)inst_mask_tensor.sizes()[0];
    int w=(int)inst_mask_tensor.sizes()[1];

    float inst_mask_area = inst_mask_tensor.sum(torch::IntArrayRef({0,1})).item().toFloat();
    cv::Point2f inst_center_pt = (instInfo.min_pt + instInfo.max_pt)/2.;

    int id_match=-1;
    float iou_max=0;
    for(const auto &[key, inst_j] : instances){
        ///根据速度计算当前的物体
        cv::Point2i delta = inst_j.box_vel * inst_j.delta_time;
        auto curr_min_pt = cv::Point2i(inst_j.box_min_pt) + delta;
        auto curr_max_pt = cv::Point2i(inst_j.box_max_pt) + delta;
        cv::Point2f curr_center_pt = (curr_max_pt+curr_min_pt)/2.;

        //cout<<fmt::format("id:{},box_vel:({},{}),delta_time:{},delta:({},{})",inst_j.id,inst_j.box_vel.x,inst_j.box_vel.y,inst_j.delta_time,delta.x,delta.y)<<endl;
        auto delta_pt = curr_center_pt - inst_center_pt;
        float delta_pt_abs = std::abs(delta_pt.x)+std::abs(delta_pt.y);

        if(getBoxIoU(instInfo.min_pt, instInfo.max_pt, curr_min_pt, curr_max_pt) > 0 || delta_pt_abs<100){
            torch::Tensor intersection_mask;
            if(delta.x>=0 && delta.y>=0){
                intersection_mask = inst_mask_tensor.index({Slice(delta.y,None),Slice(delta.x,None)}) *
                        inst_j.mask_tensor.index({Slice(None,h-delta.y),Slice(None,w-delta.x)});
            }
            else if(delta.x<0 && delta.y>=0){
                intersection_mask = inst_mask_tensor.index({Slice(delta.y,None),Slice(None,w+delta.x)}) *
                        inst_j.mask_tensor.index({Slice(None,h-delta.y),Slice(-delta.x,None)});
            }
            else if(delta.x>=0 && delta.y<0){
                intersection_mask = inst_mask_tensor.index({Slice(None,h+delta.y),Slice(delta.x,None)}) *
                        inst_j.mask_tensor.index({Slice(-delta.y,None),Slice(None,w-delta.x)});
            }
            else if(delta.x<0 && delta.y<0){
                intersection_mask = inst_mask_tensor.index({Slice(None,h+delta.y),Slice(None,w+delta.x)}) *
                        inst_j.mask_tensor.index({Slice(-delta.y,None),Slice(-delta.x,None)});
            }

            float intersection_area = intersection_mask.sum(torch::IntArrayRef({0,1})).item().toFloat();
            float iou = intersection_area / (inst_mask_area + inst_j.mask_area - intersection_area);
            //cout<<fmt::format("vel_area:{} naive:{}",intersection_area,area_test)<<endl;

            if(iou > iou_max){
                iou_max = iou;
                id_match = (int)inst_j.id;
            }
            //cout<<fmt::format("{}-{} iou:{} a:{} b:{} is:{}",instInfo.id,inst_j.id,iou,inst_mask_area,inst_j.mask_area,intersection_area)<<endl;

        }
        else{
            //cout<<fmt::format("{}-{} x",instInfo.id,inst_j.id)<<endl;

        }
    }

    return {id_match,iou_max,inst_mask_area};
}

cv::Mat InstsFeatManager::addInstances(SegImage &img)
{
    double current_time = img.time0;
    int n_inst = (int)img.insts_info.size();
    ///set inst time
    for(auto &[key,inst] : instances){
        inst.delta_time = current_time - inst.last_time;
        inst.last_time = current_time;
    }

    if(img.insts_info.empty()){
        //mask_background.release();
        mask_background_gpu.release();
        return {};
    }
    assert(img.mask_tensor.sizes()[0] == img.insts_info.size());

    cv::Size mask_size((int)img.mask_tensor.sizes()[2],(int)img.mask_tensor.sizes()[1]);
    //mask_background = img.merge_mask;
    mask_background_gpu = img.merge_mask_gpu;

    for(int i=0; i < n_inst; ++i)
    {
        auto inst_mask_tensor = img.mask_tensor[i];
        auto instInfo = img.insts_info[i];

        ///寻找匹配的实例
        auto [id_match,iou_max,inst_mask_area] = getMatchInst(instInfo,inst_mask_tensor);

        ///更新实例
        if(iou_max > 0.01 && instances[id_match].class_id == instInfo.label_id){
            //instances[id_match].mask_img = img.inst_masks[i];
            instances[id_match].mask_img_gpu = instInfo.mask_gpu;
            instances[id_match].mask_tensor = inst_mask_tensor;
            instances[id_match].box_min_pt = instInfo.min_pt;
            instances[id_match].box_max_pt = instInfo.max_pt;
            instances[id_match].mask_area = inst_mask_area;
            instances[id_match].last_frame_cnt = global_frame_id;
            if(instances[id_match].delta_time > 0){
                instances[id_match].box_vel = (instInfo.mask_center - instances[id_match].box_center_pt)/instances[id_match].delta_time;
            }
            instances[id_match].box_center_pt = instInfo.mask_center;
            //cout<<fmt::format("Update,id:{},iou_max:{}",instances[id_match].id,iou_max)<<endl;
        }
        ///创建实例
        else{
            unsigned int id=global_instance_id+i;
            InstFeat inst_feat(id, instInfo.label_id);
            //inst_feat.mask_img = img.inst_masks[i];
            inst_feat.mask_img_gpu = instInfo.mask_gpu;
            inst_feat.mask_tensor = inst_mask_tensor;
            inst_feat.box_min_pt = instInfo.min_pt;
            inst_feat.box_max_pt = instInfo.max_pt;
            inst_feat.box_center_pt = instInfo.mask_center;
            inst_feat.box_vel = cv::Point2f(0,0);
            inst_feat.class_id = instInfo.label_id;
            inst_feat.mask_area = inst_mask_area;
            inst_feat.last_time = current_time;
            inst_feat.last_frame_cnt = global_frame_id;
            instances.insert({id,inst_feat});
            //cout<<fmt::format("Insert,id:{},iou_max:{}",id,iou_max)<<endl;
        }
    }

    global_instance_id+= n_inst;
    return ~mask_background;
}



void InstsFeatManager:: addInstancesGPU(const SegImage &img)
{
    double current_time = img.time0;
    int n_inst = (int)img.insts_info.size();
    ///set inst time
    for(auto &[key,inst] : instances){
        inst.delta_time = current_time - inst.last_time;
        inst.last_time = current_time;
    }

    if(img.insts_info.empty()){
        return;
    }

    assert(img.mask_tensor.sizes()[0] == img.insts_info.size());

    cv::Size mask_size((int)img.mask_tensor.sizes()[2],(int)img.mask_tensor.sizes()[1]);
    //mask_background = img.merge_mask;

    for(int i=0; i < n_inst; ++i)
    {
        auto inst_mask_tensor = img.mask_tensor[i];
        auto instInfo = img.insts_info[i];

        ///寻找匹配的实例
        auto [id_match,iou_max,inst_mask_area] = getMatchInst(instInfo,inst_mask_tensor);

        ///更新实例
        if(iou_max > 0.01 && instances[id_match].class_id == instInfo.label_id){
            //instances[id_match].mask_img = img.inst_masks[i];
            instances[id_match].mask_img_gpu = instInfo.mask_gpu;
            instances[id_match].mask_img = instInfo.mask_cv;
            instances[id_match].mask_tensor = inst_mask_tensor;
            instances[id_match].box_min_pt = instInfo.min_pt;
            instances[id_match].box_max_pt = instInfo.max_pt;
            instances[id_match].mask_area = inst_mask_area;
            instances[id_match].last_frame_cnt = global_frame_id;
            if(instances[id_match].delta_time > 0){
                instances[id_match].box_vel = (instInfo.mask_center - instances[id_match].box_center_pt)/instances[id_match].delta_time;
            }
            instances[id_match].box_center_pt = instInfo.mask_center;
            //cout<<fmt::format("Update,id:{},iou_max:{}",instances[id_match].id,iou_max)<<endl;
        }
        ///创建实例
        else{
            unsigned int id=global_instance_id+i;
            InstFeat inst_feat(id, instInfo.label_id);
            //inst_feat.mask_img = img.inst_masks[i];
            inst_feat.mask_img_gpu = instInfo.mask_gpu;
            inst_feat.mask_img = instInfo.mask_cv;
            inst_feat.mask_tensor = inst_mask_tensor;
            inst_feat.box_min_pt = instInfo.min_pt;
            inst_feat.box_max_pt = instInfo.max_pt;
            inst_feat.box_center_pt = instInfo.mask_center;
            inst_feat.box_vel = cv::Point2f(0,0);
            inst_feat.class_id = instInfo.label_id;
            inst_feat.mask_area = inst_mask_area;
            inst_feat.last_time = current_time;
            inst_feat.last_frame_cnt = global_frame_id;
            instances.insert({id,inst_feat});
            //cout<<fmt::format("Insert,id:{},iou_max:{}",id,iou_max)<<endl;
        }
    }

    global_instance_id+= n_inst;
}



void InstsFeatManager:: addInstancesByTracking( SegImage &img)
{
    double current_time = img.time0;
    int n_inst = (int)img.insts_info.size();

    if(img.insts_info.empty()){
        return;
    }

    assert(img.mask_tensor.sizes()[0] == img.insts_info.size());

    cv::Size mask_size((int)img.mask_tensor.sizes()[2],(int)img.mask_tensor.sizes()[1]);
    //mask_background = img.merge_mask;

    auto trks = tracker->update(img.insts_info,img.color0);

    for(auto &inst : trks){
        unsigned int id=inst.track_id;
        if(instances.count(id)==0){
            InstFeat inst_feat(id, inst.label_id);
            inst_feat.mask_img_gpu = inst.mask_gpu;
            inst_feat.mask_img = inst.mask_cv;
            inst_feat.mask_tensor = inst.mask_tensor;
            inst_feat.box_min_pt = inst.min_pt;
            inst_feat.box_max_pt = inst.max_pt;
            inst_feat.box_center_pt = inst.mask_center;
            inst_feat.box_vel = cv::Point2f(0,0);
            inst_feat.class_id = inst.label_id;
            inst_feat.last_time = current_time;
            inst_feat.last_frame_cnt = global_frame_id;
            instances.insert({id,inst_feat});
            debug_t("Create inst:{} cls:{}",id,inst.name);
        }
        else{
            instances[id].mask_img_gpu = inst.mask_gpu;
            instances[id].mask_img = inst.mask_cv;
            instances[id].mask_tensor = inst.mask_tensor;
            instances[id].box_min_pt = inst.min_pt;
            instances[id].box_max_pt = inst.max_pt;
            instances[id].last_frame_cnt = global_frame_id;
            instances[id].box_center_pt = inst.mask_center;
            debug_t("Update inst:{} cls:{}",id,inst.name);
        }
    }
    global_instance_id+= n_inst;
}






vector<uchar> InstsFeatManager::rejectWithF(InstFeat &inst, int col, int row) const
{
        vector<cv::Point2f> un_cur_pts(inst.curr_points.size()), un_prev_pts(inst.last_points.size());
        for (unsigned int i = 0; i < inst.curr_points.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            camera->liftProjective(Eigen::Vector2d(inst.curr_points[i].x, inst.curr_points[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f((float)(tmp_p.x()), (float)tmp_p.y());

            camera->liftProjective(Eigen::Vector2d(inst.last_points[i].x, inst.last_points[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f((float)tmp_p.x(), (float)tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, Config::F_THRESHOLD, 0.99, status);
    return status;
}





void InstsFeatManager::ptsVelocity(double dt,vector<unsigned int> &ids,vector<cv::Point2f> &curr_un_pts,
                                   std::map<unsigned int, cv::Point2f> &prev_id_pts,std::map<unsigned int, cv::Point2f> &output_cur_id_pts,
                                   vector<cv::Point2f> &output_velocity
                                   ){
    output_velocity.clear();
    output_cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
        output_cur_id_pts.insert({ids[i], curr_un_pts[i]});

    if (!prev_id_pts.empty())
    {
        for (unsigned int i = 0; i < curr_un_pts.size(); i++){
            auto it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end()){
                double v_x = (curr_un_pts[i].x - it->second.x) / dt;
                double v_y = (curr_un_pts[i].y - it->second.y) / dt;
                output_velocity.emplace_back(v_x, v_y);
            }
            else
                output_velocity.emplace_back(0, 0);
        }
    }
    else
    {
        for (unsigned int i = 0; i < curr_un_pts.size(); i++)
            output_velocity.emplace_back(0, 0);
    }
}



void InstsFeatManager::visualizeInst(cv::Mat &img)
{
    for(const auto &[id,inst]: instances){
        if(inst.lost_num>0 || inst.curr_points.empty())
            continue;

        for(const auto &[pt1,pt2] : inst.visual_points_pair){
            //cv::circle(img, pt1, 2, cv::Scalar(255, 255, 255), 2);//上一帧的点
            cv::circle(img, pt2, 2, inst.color, 2);//当前帧的点
            //cv::arrowedLine(img, pt2, pt1, inst.color, 1, 8, 0, 0.2);
        }
        for(const auto &pt : inst.visual_new_points){
            cv::circle(img, pt, 2, inst.color, 2);
        }
        cv::rectangle(img,inst.box_min_pt,inst.box_max_pt,inst.color);
        //std::string label=fmt::format("id:{},tck:{}",id,inst.curr_points.size() - inst.visual_new_points.size());
        std::string label=fmt::format("id:{}",id);
        cv::putText(img, label, inst.feats_center_pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, inst.color, 2);

        /*if(vel_map.count(inst.id)!=0){
            auto anchor=inst.feats_center_pt;
            anchor.y += 40;
            double v_abs = vel_map[inst.id].v.norm();
            cv::putText(img, fmt::format("v:{:.2f} m/s",v_abs),anchor,cv::FONT_HERSHEY_SIMPLEX,1.0,inst.color,2);
        }*/

    }

    cv::imshow("insts",img);
    cv::waitKey(1);
}


void InstsFeatManager::drawInsts(cv::Mat& img)
{
    if(Config::SLAM == SlamType::DYNAMIC){
        for(const auto &[id,inst]: instances){
            if(inst.lost_num>0 || inst.curr_points.empty())
                continue;

            for(const auto &[pt1,pt2] : inst.visual_points_pair){
                //cv::circle(img, pt1, 2, cv::Scalar(255, 255, 255), 2);//上一帧的点
                cv::circle(img, pt2, 2, inst.color, 2);//当前帧的点
                //cv::arrowedLine(img, pt2, pt1, inst.color, 1, 8, 0, 0.2);
            }
            for(const auto &pt : inst.visual_new_points){
                cv::circle(img, pt, 3, cv::Scalar(255,255,255), 2);
            }
            std::string label=fmt::format("id:{},tck:{}",id,inst.curr_points.size() - inst.visual_new_points.size());
            //cv::putText(img, label, inst.box_center_pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, inst.color, 2);
            draw_text(img, label, inst.color, inst.box_center_pt, 1.0,2,false);

            /*if(vel_map.count(inst.id)!=0){
                auto anchor=inst.feats_center_pt;
                anchor.y += 40;
                double v_abs = vel_map[inst.id].v.norm();
                cv::putText(img, fmt::format("v:{:.2f} m/s",v_abs),anchor,cv::FONT_HERSHEY_SIMPLEX,1.0,inst.color,2);
            }*/

            if(Config::Dataset == DatasetType::KITTI){
                float rows_offset = img.rows /2;
                for(auto pt : inst.right_points){
                    pt.y+= rows_offset;
                    cv::circle(img, pt, 2, inst.color, 2);
                }
            }
            else{
                float cols_offset = img.cols /2;
                for(auto pt : inst.right_points){
                    pt.x+= cols_offset;
                    cv::circle(img, pt, 2, inst.color, 2);
                }
            }

        }


        /*for(const auto& pt : visual_new_points){
            cv::circle(img, pt, 3, cv::Scalar(0,0,255), 3);
        }*/

    }


}



