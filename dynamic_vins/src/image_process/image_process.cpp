//
// Created by chen on 2022/6/13.
//

#include "image_process.h"

#include "utils/dataset/viode_utils.h"
#include "deeplearning_utils.h"

namespace dynamic_vins{ \


ImageProcessor::ImageProcessor(const std::string &config_file,const std::string &seq_name){
    detector2d.reset(new Detector2D(config_file,seq_name));
    if(cfg::use_det3d){
        detector3d.reset(new Detector3D(config_file,seq_name));
    }
    //flow_estimator = std::make_unique<FlowEstimator>(config_file);
    stereo_matcher = std::make_shared<MyStereoMatcher>(config_file,seq_name);
}


std::vector<Box3D::Ptr> ImageProcessor::BoxAssociate2Dto3D(std::vector<Box3D::Ptr> &boxes3d,std::vector<Box2D::Ptr> &boxes2d)
{
    int target_size=boxes2d.size();
    if(target_size==0){
        return {};
    }
    std::vector<Box3D::Ptr> matched_boxes3d(target_size);


    string log_text="BoxAssociate2Dto3D:\n";

    vector<bool> match_vec(boxes3d.size(),false);

    for(int k=0;k<target_size;++k){
        auto &box2d = boxes2d[k];

        matched_boxes3d[k]= nullptr;
        vector<Box3D::Ptr> candidate_match;
        vector<int> candidate_idx;

        double max_iou=0;
        int max_idx=-1;
        for(size_t i=0;i<boxes3d.size();++i){
            if(match_vec[i])
                continue;

            cv::Rect proj_rect(boxes3d[i]->box2d.min_pt,boxes3d[i]->box2d.max_pt);
            float iou = Box2D::IoU(box2d->rect,proj_rect);

            ///类别要一致
            if(box2d->class_name != boxes3d[i]->class_name)
                continue;

            if(iou>0.1){
                candidate_match.push_back(boxes3d[i]);
                candidate_idx.push_back(i);
            }
            /*if(iou > max_iou){
                max_idx = i;
                max_iou = iou;
            }*/
        }

        double min_dist= std::numeric_limits<double>::max();
        int min_idx=-1;
        for(int i=0;i<candidate_match.size();++i){
            if(candidate_match[i]->center_pt.norm() < min_dist){
                min_dist=candidate_match[i]->center_pt.norm();
                min_idx = candidate_idx[i];
            }
        }

        //if(max_iou > 0.1){
        //    match_vec[max_idx]=true;
        //    inst.box3d = boxes[max_idx];
        //    Debugt("id:{} box2d:{} box3d:{}",inst_id,coco::CocoLabel[inst.class_id],
        //           NuScenes::GetClassName(boxes[max_idx]->class_id));
        //}

        if(!candidate_match.empty()){
            match_vec[min_idx]=true;
            matched_boxes3d[k] = boxes3d[min_idx];

            //log_text += fmt::format("result : id:{} box2d:{} box3d:{}\n",inst_id,inst.box2d->class_name,
            //                        boxes[min_idx]->class_name);
        }
    }

    Debugt(log_text);

    return matched_boxes3d;
}





void ImageProcessor::Run(SemanticImage &img) {

    TicToc tt;

    if(cfg::is_undistort_input){
        ///去畸变
        //由于需要检测直线，因此对整张图像去畸变
        cv::Mat un_image0,un_image1;
        cv::remap(img.color0, un_image0, cam_s.left_undist_map1, cam_s.left_undist_map2, CV_INTER_LINEAR);

        img.color0 = un_image0;
        if(cfg::is_stereo){
            cv::remap(img.color1, un_image1, cam_s.right_undist_map1, cam_s.right_undist_map2, CV_INTER_LINEAR);
            img.color1 = un_image1;
        }
    }

    ///rgb to gray
    tt.Tic();
    img.SetGrayImageGpu();

    ///均衡化
    //static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    //clahe->apply(img.gray0, img.gray0);
    //if(!img.gray1.empty())
    //    clahe->apply(img.gray1, img.gray1);

    if(cfg::slam == SLAM::kRaw || cfg::slam == SLAM::kRaw){
        return;
    }

    img.img_tensor = Pipeline::ImageToTensor(img.color0);

    //torch::Tensor img_clone = img_tensor.clone();
    //torch::Tensor img_tensor = Pipeline::ImageToTensor(img.color0_gpu);

    //启动光流估计线程
    //if(cfg::use_dense_flow){
        //flow_estimator->Launch(img);
    //}

    ///启动3D目标检测线程
    if(cfg::slam == SLAM::kDynamic && cfg::use_det3d){
        detector3d->Launch(img);
    }

    ///启动双目估计
    stereo_matcher->Launch(img.seq);

    Infos("ImageProcess prepare: {} ms", tt.TocThenTic());

    ///实例分割,并设置mask
    tt.Tic();
    if(cfg::slam == SLAM::kNaive || cfg::slam == SLAM::kDynamic){
        if(!cfg::is_input_seg){
            detector2d->Launch(img);

            if(cfg::slam == SLAM::kNaive)
                img.SetBackgroundMask();
            else if(cfg::slam == SLAM::kDynamic){
                img.SetMaskAndRoi();
            }
        }
        else{
            if(cfg::dataset == DatasetType::kViode){
                if(cfg::slam == SLAM::kNaive)
                    VIODE::SetViodeMaskSimple(img);
                else if(cfg::slam == SLAM::kDynamic)
                    VIODE::SetViodeMaskAndRoi(img);
            }
            else{
                std::cerr<<"ImageProcessor::Run()::set_mask not is implemented, as dataset is "<<cfg::dataset_name<<endl;
                std::terminate();
            }
        }
        Infos("ImageProcess SetMask: {} ms", tt.TocThenTic());

        string log_text="detector2d results:\n";
        for(auto &box2d:img.boxes2d){
            log_text += fmt::format("inst:{} cls:{} min_pt:({},{}),max_pt:({},{})\n",box2d->id, box2d->class_name,
                                    box2d->min_pt.x,box2d->min_pt.y,box2d->max_pt.x,box2d->max_pt.y);
        }
        Debugs(log_text);

    }


    //log
    //for(auto &inst : img.insts_info)
    //    Debugs("img.insts_info id:{} min_pt:({},{}),max_pt:({},{})",
    //    inst.id,inst.min_pt.x,inst.min_pt.y,inst.max_pt.x,inst.max_pt.y);

     //获得光流估计
     //if(cfg::use_dense_flow){
        //img.flow= flow_estimator->WaitResult();
        //if(!img.flow.data){
        //img.flow = cv::Mat(img.color0.size(),CV_32FC2,cv::Scalar_<float>(0,0));
        //}
    //}

    ///双目立体匹配得到深度
    img.disp = stereo_matcher->WaitResult();

    ///读取离线检测的3D包围框
    if(cfg::slam == SLAM::kDynamic && cfg::use_det3d){
        img.boxes3d = detector3d->WaitResult();
    }


    ///只跟踪车辆
    if(cfg::dataset == DatasetType::kKitti){
        vector<Box2D::Ptr> boxes;
        for(auto it=img.boxes2d.begin(),it_next=it;it!=img.boxes2d.end();it=it_next){
            it_next++;
            if(
                    (*it)->class_name=="Car" ||
                    (*it)->class_name=="Van" ||
                    (*it)->class_name=="Truck" ||
                    (*it)->class_name=="Tram"){
                boxes.emplace_back(*it);
            }
        }

        img.boxes2d = boxes;
    }


    ///2D-3D关联
    //img.boxes3d = BoxAssociate2Dto3D(img.boxes3d,img.boxes2d);


}











}
