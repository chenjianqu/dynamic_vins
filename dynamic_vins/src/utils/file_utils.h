/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FILE_UTILS_H
#define DYNAMIC_VINS_FILE_UTILS_H

#include <filesystem>
#include "utils/def.h"

namespace dynamic_vins{\

int CreateDirs(const std::string& dir);

void ClearDirectory(const string &path);

vector<fs::path> GetDirectoryFileNames(const string &path);

bool CheckIsDir(const string &dir);

void GetAllImageFiles(const string& dir, vector<string> &files) ;

void GetAllFiles(const string& dir, vector<string> &files,const string &filter_suffix=string()) ;

void WriteTextFile(const string& path,std::string& text);

}

#endif //DYNAMIC_VINS_FILE_UTILS_H
