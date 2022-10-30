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

/**
 * 清除某个目录下的所有文件
 * @param path
 */
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
        cout<<dir<<" not exists. Please check."<<endl;
        return false;
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



cv::Scalar BgrColor(const string &color_str,bool is_norm){
    cv::Scalar color;
    color[3]=1.;
    if(color_str=="white"){
        color[0]=1.;
        color[1]=1.;
        color[2]=1.;
    }
    else if(color_str=="black"){
        color[0]=0;
        color[1]=0;
        color[2]=0;
    }
    else if(color_str=="gray"){
        color[0]=0.5;
        color[1]=0.5;
        color[2]=0.5;
    }
    else if(color_str=="blue"){
        color[0]=1.;
        color[1]=0;
        color[2]=0;
    }
    else if(color_str=="green"){
        color[0]=0;
        color[1]=1.;
        color[2]=0;
    }
    else if(color_str=="red"){
        color[0]=0;
        color[1]=0;
        color[2]=1.;
    }
    else if(color_str=="yellow"){//红绿混合
        color[0]= 0;
        color[1]= 1;
        color[2]= 1;
    }
    else if(color_str=="cyan"){//青色,蓝绿混合
        color[0]= 1;
        color[1]= 1;
        color[2]= 0;
    }
    else if(color_str=="magenta"){//品红,红蓝混合
        color[0]= 1;
        color[1]= 0;
        color[2]= 1;
    }
    else{
        color[0]=1.;
        color[1]=1.;
        color[2]=1.;
    }

    if(!is_norm){
        color[0] = color[0] * 255;
        color[1] = color[1] * 255;
        color[2] = color[2] * 255;
    }

    return color;
}






}