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



int CreateDir(const std::string& dir)
{
    int ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (ret && errno == EEXIST){
        printf("dir[%s] already exist.\n",dir.c_str());
    }
    else if (ret){
        printf("create dir[%s] error: %d %s\n" ,dir.c_str(),ret ,strerror(errno));
        return -1;
    }
    else{
        printf("create dir[%s] success.\n", dir.c_str());
    }
    return 0;
}

std::string GetParentDir(const std::string& dir)
{
    std::string pdir = dir;
    if(pdir.length() < 1 || (pdir[0] != '/')){
        return "";
    }
    while(pdir.length() > 1 && (pdir[pdir.length() -1] == '/')) pdir = pdir.substr(0,pdir.length() -1);

    pdir = pdir.substr(0,pdir.find_last_of('/'));
    return pdir;
}

/**
 * 递归创建多级目录
 * @param dir
 * @return
 */
int CreateDirs(const std::string& dir){
    int ret = 0;
    if(dir.empty())
        return -1;
    std::string pdir;
    if((ret = CreateDir(dir)) == -1){
        pdir = GetParentDir(dir);
        if((ret = CreateDir(pdir)) == 0){
            ret = CreateDir(dir);
        }
    }
    return ret;
}




/**
 * 清除某个目录下的所有文件
 * @param path
 */
void ClearDirectory(const string &path){
    fs::path dir_path(path);
    if(!fs::exists(dir_path)){
        //int isCreate = mkdir(path.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        int isCreate = CreateDirs(path.c_str());
    }
    fs::directory_iterator dir_iter(dir_path);
    for(auto &it : dir_iter){
        remove(it.path().c_str());
    }
}

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


/**
 * 检查一个路径是否是目录
 * @param dir
 * @return
 */
bool CheckIsDir(const string &dir) {
    if (! std::filesystem::exists(dir)) {
        cout<<"CheckIsDir() "<<dir<<" not exists. Please check."<<endl;
        throw std::runtime_error(dir+" not exists. Please check.");
    }
    std::filesystem::directory_entry entry(dir);
    if (entry.is_directory())
        return true;
    return false;
}

/**
 * 递归的搜索一个目录下所有的图像文件，以 jpg,jpeg,png 结尾的文件
 * @param dir
 * @param files
 */
void GetAllImageFiles(const string& dir, vector<string> &files) {
    // 首先检查目录是否为空，以及是否是目录
    if (!CheckIsDir(dir))
        return;

    // 递归遍历所有的文件
    std::filesystem::directory_iterator iters(dir);
    for(auto &iter: iters) {
        string file_path(dir);
        file_path += "/";
        file_path += iter.path().filename();

        // 查看是否是目录，如果是目录则循环递归
        if (CheckIsDir(file_path)) {
            GetAllImageFiles(file_path, files);
        }
        //不是目录则检查后缀是否是图像
        else {
            string extension = iter.path().extension(); // 获取文件的后缀名
            if (extension == ".jpg" || extension == ".png" || extension == ".jpeg") {
                files.push_back(file_path);
            }
        }
    }
}

/**
 * 获取某个目录下的所有文件
 * @param dir 目录
 * @param files 输出的文件列表
 * @param filter_suffix 后缀，默认不设置
 */
void GetAllFiles(const string& dir, vector<string> &files,const string &filter_suffix){
    // 首先检查目录是否为空，以及是否是目录
    if (!CheckIsDir(dir))
        return;

    // 递归遍历所有的文件
    std::filesystem::directory_iterator iters(dir);
    for(auto &iter: iters) {
        string file_path(dir);
        file_path += "/";
        file_path += iter.path().filename();

        // 查看是否是目录，如果是目录则循环递归
        if (CheckIsDir(file_path)) {
            GetAllFiles(file_path, files,filter_suffix);
        }
        //不是目录则检查后缀是否是满足条件
        else {
            string extension = iter.path().extension(); // 获取文件的后缀名
            if (!filter_suffix.empty() && extension == filter_suffix) {
                files.push_back(file_path);
            }
            else{
                files.push_back(file_path);
            }
        }
    }
}



/**
 * 将字符串写入到文件中
 * @param path
 * @param text
 */
void WriteTextFile(const string& path,std::string& text){
    static std::set<string> path_set;
    ///第一次,清空文件
    if(path_set.find(path)==path_set.end()){
        path_set.insert(path);
        std::ofstream fout( path.data(), std::ios::out);
        fout.close();
    }

    ///添加到文件后面
    std::ofstream fout(path.data(), std::ios::app);
    fout<<text<<endl;
    fout.close();

}







}