/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/


#include "parameters.h"
#include <filesystem>

#include "utility/viode_utils.h"

namespace dynamic_vins{\


void InitLogger()
{
    auto resetLogFile=[](const std::string &path){
        if(!fs::exists(path)){
            std::ifstream file(path);//创建文件
            file.close();
        }
        else{
            std::ofstream file(path,std::ios::trunc);//清空文件
            file.close();
        }
    };

    auto getLogLevel=[](const std::string &level_str){
        if(level_str=="debug")
            return spdlog::level::debug;
        else if(level_str=="info")
            return spdlog::level::info;
        else if(level_str=="warn")
            return spdlog::level::warn;
        else if(level_str=="error" || level_str=="err")
            return spdlog::level::err;
        else if(level_str=="critical")
            return spdlog::level::critical;
        else{
            cerr<<"log level not right, set default warn"<<endl;
            return spdlog::level::warn;
        }
    };

    resetLogFile(Config::kEstimatorLogPath);
    vio_logger = spdlog::basic_logger_mt("estimator_log", Config::kEstimatorLogPath);
    vio_logger->set_level(getLogLevel(Config::kEstimatorLogLevel));
    vio_logger->flush_on(getLogLevel(Config::kEstimatorLogFlush));

    resetLogFile(Config::kFeatureTrackerLogPath);
    tk_logger = spdlog::basic_logger_mt("tracker_log", Config::kFeatureTrackerLogPath);
    tk_logger->set_level(getLogLevel(Config::kFeatureTrackerLogLevel));
    tk_logger->flush_on(getLogLevel(Config::kFeatureTrackerLogFlush));

    resetLogFile(Config::kSegmentorLogPath);
    sg_logger = spdlog::basic_logger_mt("segmentor_log", Config::kSegmentorLogPath);
    sg_logger->set_level(getLogLevel(Config::kSegmentorLogLevel));
    sg_logger->flush_on(getLogLevel(Config::kSegmentorLogFlush));
}

/**
 * 读取VIODE数据集的rgb_ids.txt
 * @param rgb_to_label_file
 * @return
 */
auto ReadViodeRgbIds(const string &rgb_to_label_file){
    vector<vector<int>> label_data;
    std::ifstream fp(rgb_to_label_file); //定义声明一个ifstream对象，指定文件路径
    if(!fp.is_open()){
        throw std::runtime_error(fmt::format("Can not open:{}", rgb_to_label_file));
    }
    string line;
    getline(fp,line); //跳过列名，第一行不做处理
    while (getline(fp,line)){ //循环读取每行数据
        vector<int> data_line;
        string number;
        std::istringstream read_str(line); //string数据流化
        for(int j = 0;j < 4;j++){ //可根据数据的实际情况取循环获取
            getline(read_str,number,','); //将一行数据按'，'分割
            data_line.push_back(atoi(number.c_str())); //字符串传int
        }
        label_data.push_back(data_line); //插入到vector中
    }
    fp.close();

    std::unordered_map<unsigned int,int> rgb_to_key;
    for(const auto& v : label_data){
        rgb_to_key.insert(std::make_pair(VIODE::PixelToKey(v[1], v[2], v[3]), v[0]));
    }
    return rgb_to_key;
}



Config::Config(const std::string &file_name)
{
    cv::FileStorage fs(file_name, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(fmt::format("ERROR: Wrong path to settings:{}\n",file_name));
    }

    int slam_type_index;
    fs["slam_type"]>>slam_type_index;
    if(slam_type_index==0){
        slam=SlamType::kRaw;
        cout<<"SlamType::kRaw"<<endl;
    }
    else if(slam_type_index==1){
        slam=SlamType::kNaive;
        cout<<"SlamType::kNaive"<<endl;
    }
    else{
        slam=SlamType::kDynamic;
        cout<<"SlamType::kDynamic"<<endl;
    }

    std::string dataset_type_string;
    fs["dataset_type"]>>dataset_type_string;
    std::transform(dataset_type_string.begin(),dataset_type_string.end(),dataset_type_string.begin(),::tolower);//
    if(dataset_type_string=="kitti"){
        dataset = DatasetType::kKitti;
    }
    else if(dataset_type_string=="viode"){
        dataset = DatasetType::kViode;
    }
    else{
        dataset = DatasetType::kViode;
    }
    cout<<"dataset:"<<dataset_type_string<<endl;

    fs["basic_dir"] >> kBasicDir;

    fs["estimator_log_path"] >> kEstimatorLogPath;
    kEstimatorLogPath = kBasicDir + kEstimatorLogPath;
    fs["estimator_log_level"] >> kEstimatorLogLevel;
    fs["estimator_log_flush"] >> kEstimatorLogFlush;
    fs["feature_tracker_log_path"] >> kFeatureTrackerLogPath;
    kFeatureTrackerLogPath = kBasicDir + kFeatureTrackerLogPath;
    fs["feature_tracker_log_level"] >> kFeatureTrackerLogLevel;
    fs["feature_tracker_log_flush"] >> kFeatureTrackerLogFlush;
    fs["segmentor_log_path"] >> kSegmentorLogPath;
    kSegmentorLogPath = kBasicDir + kSegmentorLogPath;
    fs["segmentor_log_level"] >> kSegmentorLogLevel;
    fs["segmentor_log_flush"] >> kSegmentorLogFlush;

    fs["fnet_onnx_path"] >> kRaftFnetOnnxPath;
    kRaftFnetOnnxPath = kBasicDir + kRaftFnetOnnxPath;
    fs["fnet_tensorrt_path"] >> kRaftFnetTensorrtPath;
    kRaftFnetTensorrtPath = kBasicDir + kRaftFnetTensorrtPath;
    fs["cnet_onnx_path"] >> kRaftCnetOnnxPath;
    kRaftCnetOnnxPath = kBasicDir + kRaftCnetOnnxPath;
    fs["cnet_tensorrt_path"] >> kRaftCnetTensorrtPath;
    kRaftCnetTensorrtPath = kBasicDir + kRaftCnetTensorrtPath;
    fs["update_onnx_path"] >> kRaftUpdateOnnxPath;
    kRaftUpdateOnnxPath = kBasicDir + kRaftUpdateOnnxPath;
    fs["update_tensorrt_path"] >> kRaftUpdateTensorrtPath;
    kRaftUpdateTensorrtPath = kBasicDir + kRaftUpdateTensorrtPath;

    fs["image0_topic"] >> kImage0Topic;
    fs["image1_topic"] >> kImage1Topic;
    fs["image0_segmentation_topic"] >> kImage0SegTopic;
    fs["image1_segmentation_topic"] >> kImage1SegTopic;

    kRow = fs["image_height"];
    kCol = fs["image_width"];

    kMaxCnt = fs["max_cnt"];
    kMaxDynamicCnt = fs["max_dynamic_cnt"];
    kMinDist = fs["min_dist"];
    kMinDynamicDist = fs["min_dynamic_dist"];
    kFThreshold = fs["F_threshold"];
    kShowTrack = fs["show_track"];
    kFlowBack = fs["flow_back"];

    kMaxSolverTime = fs["max_solver_time"];
    KNumIter = fs["max_num_iterations"];
    kMinParallax = fs["keyframe_parallax"];
    kMinParallax = kMinParallax / kFocalLength;

    is_use_imu = fs["imu"];
    cout << "USE_IMU:" << is_use_imu << endl;

    if(is_use_imu){
        fs["imu_topic"] >> kImuTopic;
        ACC_N = fs["acc_n"];
        ACC_W = fs["acc_w"];
        GYR_N = fs["gyr_n"];
        GYR_W = fs["gyr_w"];
        G.z() = fs["g_norm"];
    }

    fs["output_path"] >> kOutputFolder;
    kOutputFolder = kBasicDir+kOutputFolder;
    kVinsResultPath = kOutputFolder + "/vio.csv";

    std::ofstream fout(kVinsResultPath, std::ios::out);
    fout.close();

    /// 设置 相机到IMU的外参矩阵
    ESTIMATE_EXTRINSIC = fs["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2){
        cout<<"have no prior about extrinsic param, calibrate extrinsic param"<<endl;
        RIC.emplace_back(Eigen::Matrix3d::Identity());
        TIC.emplace_back(Eigen::Vector3d::Zero());
        kExCalibResultPath = kOutputFolder + "/extrinsic_parameter.csv";
    }
    else{
        if ( ESTIMATE_EXTRINSIC == 1){
            cout<<"Optimize extrinsic param around initial guess!"<<endl;
            kExCalibResultPath = kOutputFolder + "/extrinsic_parameter.csv";
        }
        else if (ESTIMATE_EXTRINSIC == 0){
            cout<<"fix extrinsic param"<<endl;
        }
        cv::Mat cv_T;
        fs["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.emplace_back(T.block<3, 3>(0, 0));
        TIC.emplace_back(T.block<3, 1>(0, 3));
    }

    kCamNum = fs["num_of_cam"];
    if(kCamNum != 1 && kCamNum != 2){
        throw std::runtime_error("num_of_cam should be 1 or 2");
    }

    ///读取各个相机配置文件
    auto pn = file_name.find_last_of('/');
    std::string configPath = file_name.substr(0, pn);

    std::string cam0Calib;
    fs["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    kCamPath.push_back(cam0Path);

    if(kCamNum == 2){
        is_stereo = 1;
        std::string cam1Calib;
        fs["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib;
        kCamPath.push_back(cam1Path);

        cv::Mat cv_T;
        fs["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.emplace_back(T.block<3, 3>(0, 0));
        TIC.emplace_back(T.block<3, 1>(0, 3));
    }

    fs["INIT_DEPTH"] >> kInitDepth;
    fs["BIAS_ACC_THRESHOLD"]>>BIAS_ACC_THRESHOLD;
    fs["BIAS_GYR_THRESHOLD"]>>BIAS_GYR_THRESHOLD;

    TD = fs["td"];
    is_estimate_td = fs["estimate_td"];
    if (is_estimate_td){
        cout<<"Unsynchronized sensors, online estimate time offset, initial td: "<<TD<<endl;
    }
    else{
        cout<<"Synchronized sensors, fix time offset:"<<TD<<endl;
    }

    if(!is_use_imu){
        ESTIMATE_EXTRINSIC = 0;
        is_estimate_td = 0;
        cout<<"no imu, fix extrinsic param; no time offset calibration"<<endl;
    }

    ///读取VIODE动态物体对应的Label Index
    if(dataset == DatasetType::kViode){
        cv::FileNode labelIDNode=fs["dynamic_label_id"];
        for(auto && it : labelIDNode){
            ViodeDynamicIndex.insert((int)it);
        }
        ///设置VIODE的RGB2Label
        string rgb2label_file;
        fs["rgb_to_label_file"]>>rgb2label_file;
        rgb2label_file = kBasicDir + rgb2label_file;
        ViodeKeyToIndex = ReadViodeRgbIds(rgb2label_file);
    }


    if(slam != SlamType::kRaw && dataset != DatasetType::kViode){
        fs["solo_onnx_path"] >> kDetectorOnnxPath;
        kDetectorOnnxPath = kBasicDir + kDetectorOnnxPath;
        fs["solo_serialize_path"] >> kDetectorSerializePath;
        kDetectorSerializePath = kBasicDir + kDetectorSerializePath;
        fs["SOLO_NMS_PRE"] >> kSoloNmsPre;
        fs["SOLO_MAX_PER_IMG"] >> kSoloMaxPerImg;
        fs["SOLO_NMS_KERNEL"] >> kSoloNmsKernel;
        fs["SOLO_NMS_SIGMA"] >> kSoloNmsSigma;
        fs["SOLO_SCORE_THR"] >> kSoloScoreThr;
        fs["SOLO_MASK_THR"] >> kSoloMaskThr;
        fs["SOLO_UPDATE_THR"] >> kSoloUpdateThr;
    }

    if(slam == SlamType::kDynamic && dataset != DatasetType::kViode){
        fs["extractor_model_path"] >> kExtractorModelPath;
        kExtractorModelPath = kBasicDir + kExtractorModelPath;
        fs["tracking_n_init"] >> kTrackingNInit;
        fs["tracking_max_age"] >> kTrackingMaxAge;
    }

    fs["visual_inst_duration"] >> kVisualInstDuration;

    fs.release();


    if(dataset == DatasetType::kViode && (slam == SlamType::kDynamic || Config::slam == SlamType::kNaive)){
        is_input_seg = true;
    }
    else{
        is_input_seg = false;
    }
    cout << "is_input_seg:" << is_input_seg << endl;


    std::map<int,std::string> CocoLabelMap={
            {1, "person"}, {2, "bicycle"}, {3, "car"}, {4, "motorcycle"}, {5, "airplane"},
            {6, "bus"}, {7, "train"}, {8, "truck"}, {9, "boat"}, {10, "traffic light"},
            {11, "fire hydrant"}, {13, "stop sign"}, {14, "parking meter"}, {15, "bench"},
            {16, "bird"}, {17, "cat"}, {18, "dog"}, {19, "horse"}, {20, "sheep"}, {21, "cow"},
            {22, "elephant"}, {23, "bear"}, {24, "zebra"}, {25, "giraffe"}, {27, "backpack"},
            {28, "umbrella"}, {31, "handbag"}, {32, "tie"}, {33, "suitcase"}, {34, "frisbee"},
            {35, "skis"}, {36, "snowboard"}, {37, "sports ball"}, {38, "kite"}, {39, "baseball bat"},
            {40, "baseball glove"}, {41, "skateboard"}, {42, "surfboard"}, {43, "tennis racket"},
            {44, "bottle"}, {46, "wine glass"}, {47, "cup"}, {48, "fork"}, {49, "knife"}, {50, "spoon"},
            {51, "bowl"}, {52, "banana"}, {53, "apple"}, {54, "sandwich"}, {55, "orange"},
            {56, "broccoli"}, {57, "carrot"}, {58, "hot dog"}, {59, "pizza"}, {60, "donut"},
            {61, "cake"}, {62, "chair"}, {63, "couch"}, {64, "potted plant"}, {65, "bed"}, {67, "dining table"},
            {70, "toilet"}, {72, "tv"}, {73, "laptop"}, {74, "mouse"}, {75, "remote"}, {76, "keyboard"},
            {77, "cell phone"}, {78, "microwave"}, {79, "oven"}, {80, "toaster"},{ 81, "sink"},
            {82, "refrigerator"}, {84, "book"}, {85, "clock"},{ 86, "vase"}, {87, "scissors"},
            {88, "teddy bear"}, {89, "hair drier"}, {90, "toothbrush"}
    };

    CocoLabelVector.reserve(CocoLabelMap.size());
    for(auto &pair : CocoLabelMap){
        CocoLabelVector.push_back(pair.second);
    }


    InitLogger();
}




}


