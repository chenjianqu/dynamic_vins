/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "log_utils.h"

#include <filesystem>
#include <iostream>

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
    if(fs["basic_dir"].isNone()){
        std::cerr<<"basic_dir is not configured in the config file"<<std::endl;
        std::terminate();
    }
    if(fs["log_output_path"].isNone()){
        std::cerr<<"log_output_path is not configured in the config file"<<std::endl;
        std::terminate();
    }

    std::string kBasicDir;
    fs["basic_dir"] >> kBasicDir;
    fs["log_output_path"] >> kLogOutputDir;
    kLogOutputDir = kBasicDir+ kLogOutputDir;

    std::string log_level="debug";
    std::string flush_level="warn";

    std::string path = kLogOutputDir+"log_v.txt";
    fs["estimator_log_level"] >> log_level;
    fs["estimator_log_flush"] >> flush_level;
    MyLogger::ResetLogFile(path);
    MyLogger::vio_logger = spdlog::basic_logger_mt("estimator", path);
    MyLogger::vio_logger->set_level(MyLogger::GetLogLevel(log_level));
    MyLogger::vio_logger->flush_on(MyLogger::GetLogLevel(flush_level));
    MyLogger::vio_logger->set_pattern("[%H:%M:%S.%e][%t,%L] %v");//打印格式，[时间][线程ID,日志级别] 日志内容
    std::cout<<"estimator_log path:"<<path<<std::endl;

    path = kLogOutputDir+"log_t.txt";
    fs["feature_tracker_log_level"] >> log_level;
    fs["feature_tracker_log_flush"] >> flush_level;
    MyLogger::ResetLogFile(path);
    MyLogger::tk_logger = spdlog::basic_logger_mt("tracker", path);
    MyLogger::tk_logger->set_level(MyLogger::GetLogLevel(log_level));
    MyLogger::tk_logger->flush_on(MyLogger::GetLogLevel(flush_level));
    MyLogger::tk_logger->set_pattern("[%H:%M:%S.%e][%t,%L] %v");
    std::cout<<"tracker_log path:"<<path<<std::endl;

    path = kLogOutputDir+"log_s.txt";
    fs["segmentor_log_level"] >> log_level;
    fs["segmentor_log_flush"] >> flush_level;
    MyLogger::ResetLogFile(path);
    MyLogger::sg_logger = spdlog::basic_logger_mt("segmentor", path);
    MyLogger::sg_logger->set_level(MyLogger::GetLogLevel(log_level));
    MyLogger::sg_logger->flush_on(MyLogger::GetLogLevel(flush_level));
    MyLogger::sg_logger->set_pattern("[%H:%M:%S.%e][%t,%L] %v");
    std::cout<<"segmentor_log path:"<<path<<std::endl;

    fs.release();
}


}

