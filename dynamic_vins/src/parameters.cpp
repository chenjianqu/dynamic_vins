
#include "parameters.h"
#include "estimator/dynamic.h"
#include "utils.h"
#include "utility/ViodeUtils.h"

#include <filesystem>


void initLogger()
{
    auto reset_log_file=[](const std::string &path){
        if(!fs::exists(path)){
            std::ifstream file(path);//创建文件
            file.close();
        }
        else{
            std::ofstream file(path,std::ios::trunc);//清空文件
            file.close();
        }
    };

    auto get_log_level=[](const std::string &level_str){
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


    reset_log_file(Config::ESTIMATOR_LOG_PATH);
    vioLogger = spdlog::basic_logger_mt("estimator_log", Config::ESTIMATOR_LOG_PATH);
    vioLogger->set_level(get_log_level(Config::ESTIMATOR_LOG_LEVEL)); ///设置日志级别，低于该级别将不输出
    vioLogger->flush_on(get_log_level(Config::ESTIMATOR_LOG_FLUSH));///遇到err级别，立马将缓存的日志写入

    reset_log_file(Config::FEATURE_TRACKER_LOG_PATH);
    tkLogger = spdlog::basic_logger_mt("tracker_log",Config::FEATURE_TRACKER_LOG_PATH);
    tkLogger->set_level(get_log_level(Config::FEATURE_TRACKER_LOG_LEVEL));
    tkLogger->flush_on(get_log_level(Config::FEATURE_TRACKER_LOG_FLUSH));

    reset_log_file(Config::SEGMENTOR_LOG_PATH);
    sgLogger = spdlog::basic_logger_mt("segmentor_log",Config::SEGMENTOR_LOG_PATH);
    sgLogger->set_level(get_log_level(Config::SEGMENTOR_LOG_LEVEL));
    sgLogger->flush_on(get_log_level(Config::SEGMENTOR_LOG_FLUSH));
}

/**
 * 读取VIODE数据集的rgb_ids.txt
 * @param rgb2label_file
 * @return
 */
auto ReadViodeRgbIds(const string &rgb2label_file){
    vector<vector<int>> label_data;
    std::ifstream fp(rgb2label_file); //定义声明一个ifstream对象，指定文件路径
    if(!fp.is_open()){
        throw std::runtime_error(fmt::format("Can not open:{}",rgb2label_file));
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

    std::unordered_map<unsigned int,int> RGB2Key;
    for(const auto& v : label_data){
        RGB2Key.insert(std::make_pair(VIODE::pixel2key(v[1], v[2], v[3]), v[0]));
    }
    return RGB2Key;
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
        SLAM=SlamType::RAW;
        cout<<"SlamType::RAW"<<endl;
    }
    else if(slam_type_index==1){
        SLAM=SlamType::NAIVE;
        cout<<"SlamType::NAIVE"<<endl;
    }
    else{
        SLAM=SlamType::DYNAMIC;
        cout<<"SlamType::DYNAMIC"<<endl;
    }

    std::string dataset_type_string;
    fs["dataset_type"]>>dataset_type_string;
    std::transform(dataset_type_string.begin(),dataset_type_string.end(),dataset_type_string.begin(),::tolower);//
    if(dataset_type_string=="kitti"){
        Dataset = DatasetType::KITTI;
    }
    else if(dataset_type_string=="viode"){
        Dataset = DatasetType::VIODE;
    }
    else{
        Dataset = DatasetType::VIODE;
    }
    cout<<"Dataset:"<<dataset_type_string<<endl;


    fs["estimator_log_path"]>>ESTIMATOR_LOG_PATH;
    fs["estimator_log_level"]>>ESTIMATOR_LOG_LEVEL;
    fs["estimator_log_flush"]>>ESTIMATOR_LOG_FLUSH;
    fs["feature_tracker_log_path"]>>FEATURE_TRACKER_LOG_PATH;
    fs["feature_tracker_log_level"]>>FEATURE_TRACKER_LOG_LEVEL;
    fs["feature_tracker_log_flush"]>>FEATURE_TRACKER_LOG_FLUSH;
    fs["segmentor_log_path"]>>SEGMENTOR_LOG_PATH;
    fs["segmentor_log_level"]>>SEGMENTOR_LOG_LEVEL;
    fs["segmentor_log_flush"]>>SEGMENTOR_LOG_FLUSH;

    cout<<"ESTIMATOR_LOG_PATH:"<<ESTIMATOR_LOG_PATH<<endl;
    cout<<"ESTIMATOR_LOG_LEVEL:"<<ESTIMATOR_LOG_LEVEL<<endl;
    cout<<"ESTIMATOR_LOG_FLUSH:"<<ESTIMATOR_LOG_FLUSH<<endl;
    cout<<"FEATURE_TRACKER_LOG_PATH:"<<FEATURE_TRACKER_LOG_PATH<<endl;
    cout<<"FEATURE_TRACKER_LOG_LEVEL:"<<FEATURE_TRACKER_LOG_LEVEL<<endl;
    cout<<"FEATURE_TRACKER_LOG_FLUSH:"<<FEATURE_TRACKER_LOG_FLUSH<<endl;
    cout<<"SEGMENTOR_LOG_PATH:"<<SEGMENTOR_LOG_PATH<<endl;
    cout<<"SEGMENTOR_LOG_LEVEL:"<<SEGMENTOR_LOG_LEVEL<<endl;
    cout<<"SEGMENTOR_LOG_FLUSH:"<<SEGMENTOR_LOG_FLUSH<<endl;

    fs["image0_topic"] >> IMAGE0_TOPIC;
    fs["image1_topic"] >> IMAGE1_TOPIC;
    fs["image0_segmentation_topic"] >>IMAGE0_SEGMENTATION_TOPIC;
    fs["image1_segmentation_topic"] >>IMAGE1_SEGMENTATION_TOPIC;

    cout<<"IMAGE0_TOPIC:"<<IMAGE0_TOPIC<<endl;
    cout<<"IMAGE1_TOPIC:"<<IMAGE1_TOPIC<<endl;
    cout<<"IMAGE0_SEGMENTATION_TOPIC:"<<IMAGE0_SEGMENTATION_TOPIC<<endl;
    cout<<"IMAGE1_SEGMENTATION_TOPIC:"<<IMAGE1_SEGMENTATION_TOPIC<<endl;


    ROW = fs["image_height"];
    COL = fs["image_width"];
    cout<<fmt::format("ROW:{},COL:{}",ROW,COL)<<endl;


    MAX_CNT = fs["max_cnt"];
    MAX_DYNAMIC_CNT = fs["max_dynamic_cnt"];
    MIN_DIST = fs["min_dist"];
    MIN_DYNAMIC_DIST = fs["min_dynamic_dist"];
    F_THRESHOLD = fs["F_threshold"];
    SHOW_TRACK = fs["show_track"];
    FLOW_BACK = fs["flow_back"];

    SOLVER_TIME = fs["max_solver_time"];
    NUM_ITERATIONS = fs["max_num_iterations"];
    MIN_PARALLAX = fs["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    USE_IMU = fs["imu"];
    cout<<"USE_IMU:"<<USE_IMU<<endl;

    if(USE_IMU){
        fs["imu_topic"] >> IMU_TOPIC;
        cout<<"IMU_TOPIC:"<<IMU_TOPIC<<endl;
        ACC_N = fs["acc_n"];
        ACC_W = fs["acc_w"];
        GYR_N = fs["gyr_n"];
        GYR_W = fs["gyr_w"];
        G.z() = fs["g_norm"];
    }

    fs["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    cout<<"OUTPUT_FOLDER:"<<OUTPUT_FOLDER<<endl;
    cout<<"VINS_RESULT_PATH:"<<VINS_RESULT_PATH<<endl;

    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    /// 设置 相机到IMU的外参矩阵
    ESTIMATE_EXTRINSIC = fs["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2){
        cout<<"have no prior about extrinsic param, calibrate extrinsic param"<<endl;
        RIC.emplace_back(Eigen::Matrix3d::Identity());
        TIC.emplace_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else{
        if ( ESTIMATE_EXTRINSIC == 1){
            cout<<"Optimize extrinsic param around initial guess!"<<endl;
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
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


    NUM_OF_CAM = fs["num_of_cam"];
    cout<<"NUM_OF_CAM:"<<NUM_OF_CAM<<endl;
    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2){
        throw std::runtime_error("num_of_cam should be 1 or 2");
    }

    ///读取各个相机配置文件
    auto pn = file_name.find_last_of('/');
    std::string configPath = file_name.substr(0, pn);

    std::string cam0Calib;
    fs["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if(NUM_OF_CAM == 2){
        STEREO = 1;
        std::string cam1Calib;
        fs["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib;
        CAM_NAMES.push_back(cam1Path);

        cv::Mat cv_T;
        fs["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.emplace_back(T.block<3, 3>(0, 0));
        TIC.emplace_back(T.block<3, 1>(0, 3));
    }

    fs["INIT_DEPTH"]>>INIT_DEPTH;
    fs["BIAS_ACC_THRESHOLD"]>>BIAS_ACC_THRESHOLD;
    fs["BIAS_GYR_THRESHOLD"]>>BIAS_GYR_THRESHOLD;

    TD = fs["td"];
    ESTIMATE_TD = fs["estimate_td"];
    if (ESTIMATE_TD){
        cout<<"Unsynchronized sensors, online estimate time offset, initial td: "<<TD<<endl;
    }
    else{
        cout<<"Synchronized sensors, fix time offset:"<<TD<<endl;
    }


    if(!USE_IMU){
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        cout<<"no imu, fix extrinsic param; no time offset calibration"<<endl;
    }

    ///读取VIODE动态物体对应的Label Index
    if(Dataset == DatasetType::VIODE){
        cv::FileNode labelIDNode=fs["dynamic_label_id"];
        for(auto && it : labelIDNode){
            VIODE_DynamicIndex.insert((int)it);
        }
        ///设置VIODE的RGB2Label
        string rgb2label_file;
        fs["rgb_to_label_file"]>>rgb2label_file;
        VIODE_Key2Index = ReadViodeRgbIds(rgb2label_file);
    }


    if(SLAM != SlamType::RAW && Dataset!=DatasetType::VIODE){
        fs["onnx_path"] >> DETECTOR_ONNX_PATH;
        fs["serialize_path"] >> DETECTOR_SERIALIZE_PATH;

        fs["SOLO_NMS_PRE"]>>SOLO_NMS_PRE;
        fs["SOLO_MAX_PER_IMG"]>>SOLO_MAX_PER_IMG;
        fs["SOLO_NMS_KERNEL"]>>SOLO_NMS_KERNEL;
        fs["SOLO_NMS_SIGMA"]>>SOLO_NMS_SIGMA;
        fs["SOLO_SCORE_THR"]>>SOLO_SCORE_THR;
        fs["SOLO_MASK_THR"]>>SOLO_MASK_THR;
        fs["SOLO_UPDATE_THR"]>>SOLO_UPDATE_THR;
    }

    if(SLAM==SlamType::DYNAMIC && Dataset!=DatasetType::VIODE){
        fs["EXTRACTOR_MODEL_PATH"]>>EXTRACTOR_MODEL_PATH;
        fs["TRACKING_N_INIT"]>>TRACKING_N_INIT;
        fs["TRACKING_MAX_AGE"]>>TRACKING_MAX_AGE;
    }

    fs["VISUAL_INST_DURATION"]>>VISUAL_INST_DURATION;

    fs.release();


    if(Dataset == DatasetType::VIODE && ( SLAM == SlamType::DYNAMIC || Config::SLAM == SlamType::NAIVE)){
        isInputSeg = true;
    }
    else{
        isInputSeg = false;
    }
    cout<<"isInputSeg:"<<isInputSeg<<endl;

    CocoLabelVector.reserve(CocoLabelMap.size());
    for(auto &pair : CocoLabelMap){
        CocoLabelVector.push_back(pair.second);
    }


    initLogger();
}




