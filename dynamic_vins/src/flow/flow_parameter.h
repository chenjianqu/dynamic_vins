//
// Created by chen on 2022/4/25.
//

#ifndef DYNAMIC_VINS_FLOW_PARAMETER_H
#define DYNAMIC_VINS_FLOW_PARAMETER_H

#include <string>

namespace dynamic_vins{\

class FlowParameter{
public:
    inline static std::string kRaftFnetOnnxPath;
    inline static std::string kRaftFnetTensorrtPath;
    inline static std::string kRaftCnetOnnxPath;
    inline static std::string kRaftCnetTensorrtPath;
    inline static std::string kRaftUpdateOnnxPath;
    inline static std::string kRaftUpdateTensorrtPath;

    inline static std::string kFlowPreprocessPath;//预计算的光流路径

    static void SetParameters(const std::string &config_path);
};

using flow_para = FlowParameter;



}



#endif //DYNAMIC_VINS_FLOW_PARAMETER_H
