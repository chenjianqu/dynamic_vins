/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_LOG_UTILS_H
#define DYNAMIC_VINS_LOG_UTILS_H

#include <memory>
#include <string>

#include <spdlog/spdlog.h>

namespace dynamic_vins{\


class MyLogger{
public:
    static void ResetLogFile(const std::string &path);

    static spdlog::level::level_enum GetLogLevel(const std::string &level_str);

    static void InitLogger(const std::string &config_path);

    inline static std::shared_ptr<spdlog::logger> vio_logger;
    inline static std::shared_ptr<spdlog::logger> tk_logger;
    inline static std::shared_ptr<spdlog::logger> sg_logger;
};


template <typename Arg1, typename... Args>
inline void Debugv(const char* fmt, const Arg1 &arg1, const Args&... args){ MyLogger::vio_logger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void Debugv(const T& msg){MyLogger::vio_logger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void Infov(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::vio_logger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void Infov(const T& msg){MyLogger::vio_logger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void Warnv(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::vio_logger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void Warnv(const T& msg){MyLogger::vio_logger->log(spdlog::level::warn, msg);}
template <typename Arg1, typename... Args>
inline void Errorv(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::vio_logger->log(spdlog::level::err, fmt, arg1, args...);}
template<typename T>
inline void Errorv(const T& msg){MyLogger::vio_logger->log(spdlog::level::err, msg);}
template <typename Arg1, typename... Args>
inline void Criticalv(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::vio_logger->log(spdlog::level::critical, fmt, arg1, args...);}
template<typename T>
inline void Criticalv(const T& msg){MyLogger::vio_logger->log(spdlog::level::critical, msg);}

template <typename Arg1, typename... Args>
inline void Debugs(const char* fmt, const Arg1 &arg1, const Args&... args){ MyLogger::sg_logger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void Debugs(const T& msg){MyLogger::sg_logger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void Infos(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::sg_logger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void Infos(const T& msg){MyLogger::sg_logger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void Warns(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::sg_logger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void Warns(const T& msg){MyLogger::sg_logger->log(spdlog::level::warn, msg);}
template <typename Arg1, typename... Args>
inline void Errors(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::sg_logger->log(spdlog::level::err, fmt, arg1, args...);}
template<typename T>
inline void Errors(const T& msg){MyLogger::sg_logger->log(spdlog::level::err, msg);}
template <typename Arg1, typename... Args>
inline void Criticals(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::sg_logger->log(spdlog::level::critical, fmt, arg1, args...);}
template<typename T>
inline void Criticals(const T& msg){MyLogger::sg_logger->log(spdlog::level::critical, msg);}


template <typename Arg1, typename... Args>
inline void Debugt(const char* fmt, const Arg1 &arg1, const Args&... args){ MyLogger::tk_logger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void Debugt(const T& msg){MyLogger::tk_logger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void Infot(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::tk_logger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void Infot(const T& msg){MyLogger::tk_logger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void Warnt(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::tk_logger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void Warnt(const T& msg){MyLogger::tk_logger->log(spdlog::level::warn, msg);}
template <typename Arg1, typename... Args>
inline void Errort(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::tk_logger->log(spdlog::level::err, fmt, arg1, args...);}
template<typename T>
inline void Errort(const T& msg){MyLogger::tk_logger->log(spdlog::level::err, msg);}
template <typename Arg1, typename... Args>
inline void Criticalt(const char* fmt, const Arg1 &arg1, const Args&... args){MyLogger::tk_logger->log(spdlog::level::critical, fmt, arg1, args...);}
template<typename T>
inline void Criticalt(const T& msg){MyLogger::tk_logger->log(spdlog::level::critical, msg);}


}

#endif //DYNAMIC_VINS_LOG_UTILS_H
