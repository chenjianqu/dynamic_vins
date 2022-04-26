//
// Created by chen on 2022/4/25.
//

#ifndef DYNAMIC_VINS_TRACKING_PARAMETER_H
#define DYNAMIC_VINS_TRACKING_PARAMETER_H

#include <string>

namespace dynamic_vins{\

class TrackParameter{
public:
    inline static std::string kExtractorModelPath;
    inline static int kTrackingMaxAge;
    inline static int kTrackingNInit;


    static void SetParameters(const std::string &config_path);
};

using track_para = TrackParameter;





}



#endif //DYNAMIC_VINS_TRACKING_PARAMETER_H
