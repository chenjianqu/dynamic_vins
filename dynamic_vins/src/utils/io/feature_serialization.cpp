//
// Created by chen on 2022/10/29.
//

#include "feature_serialization.h"


namespace dynamic_vins{\




string Vec7dToStr(const Eigen::Matrix<double, 7, 1> &v){
    return fmt::format("{} {} {} {} {} {} {}",
                       v(0,0),v(1,0),v(2,0),v(3,0),
                       v(4,0),v(5,0),v(6,0));
}

string LineToStr(const Line &line){
    return fmt::format("{} {} {} {}",line.StartPt.x,line.StartPt.y,line.EndPt.x,line.EndPt.y);
}

/**
 * 序列化点特征到txt文件
 * @param path
 * @param points
 */
void SerializePointFeature(const string& path,const std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &points){
    std::ofstream fout(path.data(), std::ios::out);

    for(auto &[id,vec_feat] : points){
        if(vec_feat.size()==1){
            fout<<fmt::format("0 {} {}",id, Vec7dToStr(vec_feat[0].second))<<endl;
        }
        else{
            fout<<fmt::format("1 {} {} {}",id, Vec7dToStr(vec_feat[0].second),Vec7dToStr(vec_feat[1].second))<<endl;
        }
    }

    fout.close();
}

/**
 * 从txt文件中反序列点特征
 * @param path
 * @return
 */
std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
DeserializePointFeature(const string& path){
    std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;

    std::ifstream fin(path.data(), std::ios::in);
    string line;
    std::vector<std::string> tokens;
    Vec7d v;
    while(getline(fin,line)){
        tokens.clear();
        split(line, tokens, " ");
        int id = std::stoi(tokens[1]);

        v<<std::stod(tokens[2]),std::stod(tokens[3]),std::stod(tokens[4]),
        std::stod(tokens[5]),std::stod(tokens[6]),std::stod(tokens[7]),std::stod(tokens[8]);
        points[id].push_back({0,v});

        if(tokens[0]=="1"){//双目观测
            v<<std::stod(tokens[9]),std::stod(tokens[10]),std::stod(tokens[11]),
            std::stod(tokens[12]),std::stod(tokens[13]),std::stod(tokens[14]),std::stod(tokens[15]);
            points[id].push_back({1,v});
        }
    }
    fin.close();
    return points;
}

/**
 * 序列化线特征
 * @param path
 * @param lines
 */
void SerializeLineFeature(const string& path,const std::map<unsigned int, std::vector<std::pair<int,Line>>> &lines){
    std::ofstream fout(path.data(), std::ios::out);

    for(auto &[id,vec_feat] : lines){
        if(vec_feat.size()==1){
            fout<<fmt::format("0 {} {}",id, LineToStr(vec_feat[0].second))<<endl;
        }
        else{
            fout<<fmt::format("1 {} {} {}",id, LineToStr(vec_feat[0].second),LineToStr(vec_feat[1].second))<<endl;
        }
    }

    fout.close();
}

/**
 * 反序列化线特征
 * @param path
 * @return
 */
std::map<unsigned int, std::vector<std::pair<int,Line>>>
DeserializeLineFeature(const string& path){
    std::map<unsigned int, std::vector<std::pair<int,Line>>> lines;

    std::ifstream fin(path.data(), std::ios::in);
    string line;
    std::vector<std::string> tokens;
    while(getline(fin,line)){
        tokens.clear();
        split(line, tokens, " ");
        int id = std::stoi(tokens[1]);

        Line l;
        l.StartPt.x=std::stof(tokens[2]);
        l.StartPt.y=std::stof(tokens[3]);
        l.EndPt.x=std::stof(tokens[4]);
        l.EndPt.y=std::stof(tokens[5]);
        l.id = id;

        lines[id].push_back({0,l});

        if(tokens[0]=="1"){//双目观测
            Line l2;
            l2.StartPt.x=std::stof(tokens[6]);
            l2.StartPt.y=std::stof(tokens[7]);
            l2.EndPt.x=std::stof(tokens[8]);
            l2.EndPt.y=std::stof(tokens[9]);
            l2.id = id;
            lines[id].push_back({1,l2});
        }
    }
    fin.close();

    return lines;
}




}
