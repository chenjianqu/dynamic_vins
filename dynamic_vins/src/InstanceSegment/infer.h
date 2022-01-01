//
// Created by chen on 2021/11/7.
//

#ifndef DYNAMIC_VINS_INFER_H
#define DYNAMIC_VINS_INFER_H

#include <optional>
#include <memory>

#include "../featureTracker/SegmentImage.h"

#include "common.h"
#include "pipeline.h"
#include "solo.h"
#include "buffer.h"

#include <NvInfer.h>



struct InferDeleter{
    template <typename T>
    void operator()(T* obj) const{
        if (obj)
            obj->destroy();
    }
};

class Infer {
public:
    using Ptr = std::shared_ptr<Infer>;
    Infer();
    std::tuple<std::vector<cv::Mat>,std::vector<InstInfo> > forward(cv::Mat &img);
    void forward_tensor(cv::Mat &img,torch::Tensor &mask_tensor,std::vector<InstInfo> &insts);
    void forward_tensor(cv::cuda::GpuMat &img,torch::Tensor &mask_tensor,std::vector<InstInfo> &insts);

    void visualizeResult(cv::Mat &input,cv::Mat &mask,std::vector<InstInfo> &insts);

    void push_back(SegImage& img){
        std::unique_lock<std::mutex> lock(queueMutex);
        if(seg_img_list.size()< INFER_IMAGE_LIST_SIZE){
            seg_img_list.push_back(img);
        }
        queueCond.notify_one();
    }

    int get_queue_size(){
        std::unique_lock<std::mutex> lock(queueMutex);
        return (int)seg_img_list.size();
    }

    std::optional<SegImage> wait_for_result() {
        std::unique_lock<std::mutex> lock(queueMutex);
        if(!queueCond.wait_for(lock,30ms,[&]{return !seg_img_list.empty();}))
            return std::nullopt;
        //queueCond.wait(lock,[&]{return !seg_img_list.empty();});
        SegImage frame=std::move(seg_img_list.front());
        seg_img_list.pop_front();

        return frame;
    }



private:
    MyBuffer::Ptr buffer;
    Pipeline::Ptr pipeline;
    Solov2::Ptr solo;

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<IExecutionContext, InferDeleter> context;

    double infer_time{0};

    std::mutex queueMutex;
    std::condition_variable queueCond;
    std::list<SegImage> seg_img_list;
};


#endif //DYNAMIC_VINS_INFER_H
