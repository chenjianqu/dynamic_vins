//
// Created by chen on 2021/11/7.
//

#ifndef DYNAMIC_VINS_SOLO_H
#define DYNAMIC_VINS_SOLO_H

#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torchvision/vision.h>

#include "../parameters.h"
#include "../featureTracker/SegmentImage.h"


class Solov2 {
public:
    using Ptr=std::shared_ptr<Solov2>;
    Solov2(){
        size_trans=torch::from_blob(SOLO_NUM_GRIDS.data(),{int(SOLO_NUM_GRIDS.size())},torch::kFloat).clone();
        size_trans=size_trans.pow(2).cumsum(0);
    }
    static torch::Tensor MatrixNMS(torch::Tensor &seg_masks,torch::Tensor &cate_labels,torch::Tensor &cate_scores,torch::Tensor &sum_mask);

    cv::Mat getSingleSeg(std::vector<torch::Tensor> &outputs,torch::Device device,std::vector<InstInfo> &insts);
    std::tuple<std::vector<cv::Mat>,std::vector<InstInfo>> getSingleSeg(std::vector<torch::Tensor> &outputs,ImageInfo& img_info);
    void getSegTensor(std::vector<torch::Tensor> &outputs,ImageInfo& img_info,torch::Tensor &mask_tensor,std::vector<InstInfo> &insts);

    bool isResized{true};
    bool output_split_mask{true};

private:
    torch::Tensor size_trans;
};


#endif //DYNAMIC_VINS_SOLO_H
