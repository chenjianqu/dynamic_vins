//
// Created by chen on 2021/11/30.
//

#include "Extractor.h"
#include "../parameters.h"
#include "../utils.h"

namespace nn=torch::nn;


namespace {
    struct BasicBlockImpl : nn::Module {
        explicit BasicBlockImpl(int64_t c_in, int64_t c_out, bool is_downsample = false) {
            conv = register_module(
                    "conv",
                    nn::Sequential(
                            nn::Conv2d(nn::Conv2dOptions(c_in, c_out, 3)
                            .stride(is_downsample ? 2 : 1)
                            .padding(1).bias(false)),
                            nn::BatchNorm2d(c_out),
                            nn::Functional(torch::relu),
                            nn::Conv2d(nn::Conv2dOptions(c_out, c_out, 3)
                            .stride(1).padding(1).bias(false)),
                            nn::BatchNorm2d(c_out)));

            if (is_downsample) {
                downsample = register_module(
                        "downsample",
                        nn::Sequential(nn::Conv2d(nn::Conv2dOptions(c_in, c_out, 1)
                        .stride(2).bias(false)),
                                       nn::BatchNorm2d(c_out)));
            } else if (c_in != c_out) {
                downsample = register_module(
                        "downsample",
                        nn::Sequential(nn::Conv2d(nn::Conv2dOptions(c_in, c_out, 1)
                        .stride(1).bias(false)),
                                       nn::BatchNorm2d(c_out)));
            }
        }

        torch::Tensor forward(torch::Tensor x) {
            auto y = conv->forward(x);
            if (!downsample.is_empty()) {
                x = downsample->forward(x);
            }
            return torch::relu(x + y);
        }

        nn::Sequential conv{nullptr}, downsample{nullptr};
    };

    TORCH_MODULE(BasicBlock);


    nn::Sequential make_layers(int64_t c_in, int64_t c_out, size_t repeat_times, bool is_downsample = false) {
        nn::Sequential ret;
        for (size_t i = 0; i < repeat_times; ++i) {
            ret->push_back(BasicBlock(i == 0 ? c_in : c_out, c_out, i == 0 ? is_downsample : false));
        }
        return ret;
    }
}



ExtractorNetImpl::ExtractorNetImpl() {
    conv1 = nn::Module::register_module(
            "conv1",
            nn::Sequential(
                    nn::Conv2d(nn::Conv2dOptions(3,64,3).stride(1).padding(1)),
                    nn::BatchNorm2d(64),
                    nn::Functional(torch::relu)
                    ));
    conv2 = nn::Module::register_module("conv2",nn::Sequential());
    conv2->extend(*make_layers(64,64,2,false));
    conv2->extend(*make_layers(64,128,2,true));
    conv2->extend(*make_layers(128,256,2,true));
    conv2->extend(*make_layers(256,512,2,true));
}


void ExtractorNetImpl::load_form(const std::string &bin_path) {
    auto load_tensor = [](torch::Tensor t, std::ifstream &fs) {
        fs.read(static_cast<char *>(t.data_ptr()), t.numel() * sizeof(float));
    };

    auto load_Conv2d = [&](nn::Conv2d m, std::ifstream &fs) {
        load_tensor(m->weight, fs);
        if (m->options.bias()) {
            load_tensor(m->bias, fs);
        }
    };

    auto load_BatchNorm=[&](nn::BatchNorm2d m, std::ifstream &fs) {
        load_tensor(m->weight, fs);
        load_tensor(m->bias, fs);
        load_tensor(m->running_mean, fs);
        load_tensor(m->running_var, fs);
    };

    auto load_sequential=[&](nn::Sequential s, std::ifstream &fs){
        if (s.is_empty()) return;
        for (auto &m:s->children()) {
            if (auto c = std::dynamic_pointer_cast<nn::Conv2dImpl>(m)) {
                load_Conv2d(c, fs);
            } else if (auto b = std::dynamic_pointer_cast<nn::BatchNorm2dImpl>(m)) {
                load_BatchNorm(b, fs);
            }
        }
    };


    std::ifstream file(bin_path,std::ios_base::binary);
    if(!file.is_open()){
        auto msg=fmt::format("Can not open the file:{}",bin_path);
        vioLogger->critical(msg);
        throw std::runtime_error(msg);
    }
    load_sequential(conv1,file);
    for(auto &m : conv2->children()){
        auto b= std::static_pointer_cast<BasicBlockImpl>(m);
        load_sequential(b->conv,file);
        load_sequential(b->downsample,file);
    }
    file.close();
}


torch::Tensor ExtractorNetImpl::forward(torch::Tensor x) {
    //tkLogger->debug("ExtractorNetImpl::forward input sizes: {}",dims2str(x.sizes()));
    x = conv1->forward(x);
    //tkLogger->debug("ExtractorNetImpl::forward conv1 sizes: {}",dims2str(x.sizes()));
    x=torch::max_pool2d(x,3,2,1);
    //tkLogger->debug("ExtractorNetImpl::forward max_pool2d sizes: {}",dims2str(x.sizes()));
    x=conv2->forward(x);
    //tkLogger->debug("ExtractorNetImpl::forward conv2 sizes: {}",dims2str(x.sizes()));
    x=torch::avg_pool2d(x,{8,4},1);
    //tkLogger->debug("ExtractorNetImpl::forward avg_pool2d sizes: {}",dims2str(x.sizes()));
    x = x.view({x.size(0),-1});
    //tkLogger->debug("ExtractorNetImpl::forward view sizes: {}",dims2str(x.sizes()));
    x.div_(x.norm(2,1,true));
    return x;
}





Extractor::Extractor(const std::string &param_path) {
    model->load_form(param_path);
    model->to(torch::kCUDA);
    model->eval();
}


/**
 * 抽取ROI的特征
 * @param input 输入的ROI的集合
 * @return
 */
torch::Tensor Extractor::extract(std::vector<cv::Mat> &input) {
    if (input.empty()) {
        return torch::empty({0, 512});
    }

    tkLogger->debug("input size:{}",input.size());


    torch::NoGradGuard no_grad;

    static const auto MEAN = torch::tensor({0.485f, 0.456f, 0.406f}).view({1, -1, 1, 1}).cuda();
    static const auto STD = torch::tensor({0.229f, 0.224f, 0.225f}).view({1, -1, 1, 1}).cuda();

    std::vector<torch::Tensor> resized;
    for (auto &x:input) {
        cv::resize(x, x, {64, 128});
        cv::cvtColor(x, x, cv::COLOR_RGB2BGR);
        x.convertTo(x, CV_32F, 1.0 / 255);
        resized.push_back(torch::from_blob(x.data, {128, 64, 3}));
    }
    auto tensor = torch::stack(resized).cuda().permute({0, 3, 1, 2}).sub_(MEAN).div_(STD);
    return model->forward(tensor);
}

