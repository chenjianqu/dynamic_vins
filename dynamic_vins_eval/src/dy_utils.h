//
// Created by chen on 2022/11/20.
//

#ifndef DYNAMIC_VINS_EVAL_DY_UTILS_H
#define DYNAMIC_VINS_EVAL_DY_UTILS_H

#include <filesystem>

#include "def.h"

vector<fs::path> GetDirectoryFileNames(const string &path);


inline void split(const std::string& source, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
    std::regex re(delimiters);
    std::copy(std::sregex_token_iterator(source.begin(), source.end(),re,-1),
              std::sregex_token_iterator(),
              std::back_inserter(tokens));
}


#endif //DYNAMIC_VINS_EVAL_DY_UTILS_H
