//
// Created by chen on 2022/11/20.
//

#include "dy_utils.h"


/**
 * 获取目录下的所有文件名
 * @param path
 * @return
 */
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




