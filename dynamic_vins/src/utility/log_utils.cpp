//
// Created by chen on 2022/4/25.
//
#include "log_utils.h"

#include <filesystem>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>


namespace dynamic_vins{\

void MyLogger::ResetLogFile(const std::string &path){
    if(!std::filesystem::exists(path)){
        std::ifstream file(path);//创建文件
        file.close();
    }
    else{
        std::ofstream file(path,std::ios::trunc);//清空文件
        file.close();
    }
}


spdlog::level::level_enum MyLogger::GetLogLevel(const std::string &level_str){
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
        std::cerr<<"log level not right, set default warn"<<std::endl;
        return spdlog::level::warn;
    }
}

void MyLogger::InitLogger(const std::string &config_path){
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(fmt::format("ERROR: Wrong path to settings:{} ",config_path));
    }

    std::string kBasicDir;
    fs["basic_dir"] >> kBasicDir;

    std::string path;
    std::string log_level="debug";
    std::string flush_level="warn";

    fs["estimator_log_path"] >> path;
    path = kBasicDir + path;
    fs["estimator_log_level"] >> log_level;
    fs["estimator_log_flush"] >> flush_level;
    MyLogger::ResetLogFile(path);
    MyLogger::vio_logger = spdlog::basic_logger_mt("estimator", path);
    MyLogger::vio_logger->set_level(MyLogger::GetLogLevel(log_level));
    MyLogger::vio_logger->flush_on(MyLogger::GetLogLevel(flush_level));

    fs["feature_tracker_log_path"] >> path;
    path = kBasicDir + path;
    fs["feature_tracker_log_level"] >> log_level;
    fs["feature_tracker_log_flush"] >> flush_level;
    MyLogger::ResetLogFile(path);
    MyLogger::tk_logger = spdlog::basic_logger_mt("tracker", path);
    MyLogger::tk_logger->set_level(MyLogger::GetLogLevel(log_level));
    MyLogger::tk_logger->flush_on(MyLogger::GetLogLevel(flush_level));

    fs["segmentor_log_path"] >> path;
    path = kBasicDir + path;
    fs["segmentor_log_level"] >> log_level;
    fs["segmentor_log_flush"] >> flush_level;
    MyLogger::ResetLogFile(path);
    MyLogger::sg_logger = spdlog::basic_logger_mt("segmentor", path);
    MyLogger::sg_logger->set_level(MyLogger::GetLogLevel(log_level));
    MyLogger::sg_logger->flush_on(MyLogger::GetLogLevel(flush_level));

    fs.release();
}


}

