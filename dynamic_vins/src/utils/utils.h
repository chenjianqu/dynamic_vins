/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_UTILS_H
#define DYNAMIC_VINS_UTILS_H

#include <string>
#include <vector>
#include <regex>

inline void split(const std::string& source, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
    std::regex re(delimiters);
    std::copy(std::sregex_token_iterator(source.begin(), source.end(),re,-1),
              std::sregex_token_iterator(),
              std::back_inserter(tokens));
}


#endif //DYNAMIC_VINS_UTILS_H
