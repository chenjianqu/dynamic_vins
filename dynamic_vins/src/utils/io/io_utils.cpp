/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "io_utils.h"


namespace dynamic_vins{\

void ClearDirectory(const string &path){
    fs::path dir_path(path);
    if(!fs::exists(dir_path)){
        int isCreate = mkdir(path.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    }
    fs::directory_iterator dir_iter(dir_path);
    for(auto &it : dir_iter){
        remove(it.path().c_str());
    }
}


vector<fs::path> GetDirectoryFileNames(const string &path){
    vector<fs::path> names;
    fs::path dir_path(path);
    if(!fs::exists(dir_path))
        return {};
    fs::directory_iterator dir_iter(dir_path);
    for(auto &it : dir_iter)
        names.emplace_back(it.path().filename());
    std::sort(names.begin(),names.end());
    return names;
}




}