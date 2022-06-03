#include <cstdio>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <list>
#include <thread>
#include <regex>
#include<filesystem>

#include <ros/ros.h>



using namespace std;

std::string PadNumber(int number,int name_width){
    std::stringstream ss;
    ss<<std::setw(name_width)<<std::setfill('0')<<number;
    string target_name;
    ss >> target_name;
    return target_name;
}

inline void split(const std::string& source, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
    std::regex re(delimiters);
    std::copy(std::sregex_token_iterator(source.begin(), source.end(),re,-1),
              std::sregex_token_iterator(),
              std::back_inserter(tokens));
}

int main(int argc, char** argv)
{
    if(argc!=3){
        cerr<<"format:rosrun dynamic_vins_eval convert_tracking_to_object tracking_label_path save_object_label_path"<<endl;
    }

    setlocale(LC_ALL, "");//防止中文乱码
    ros::init(argc, argv, "convert_tracking_to_object");
    ros::start();

    ros::NodeHandle nh;

    string tracking_label_path = argv[1];
    string save_object_label_path= argv[2];
    save_object_label_path+="/";


    ifstream fp(tracking_label_path); //定义声明一个ifstream对象，指定文件路径
    if(!fp.is_open()){
        cerr<<"Can not open file:"<<tracking_label_path<<endl;
        return -1;
    }

    int curr_frame=-1;
    ofstream fout;

    string line;
    while (getline(fp,line)){ //循环读取每行数据
        vector<string> tokens;
        split(line,tokens," ");
        int frame=std::stoi(tokens[0]);
        cout<<frame<<endl;

        /**
    Kitti Tracking format:
   #Values    Name      Description
   1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Integer (0,1,2) indicating the level of truncation.
                     Note that this is in contrast to the object detection
                     benchmark where truncation is a float in [0,1].
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
        */


        if(curr_frame!=frame){
            fout.close();

            string out_path=save_object_label_path + PadNumber(frame,6)+".txt";
            fout.open(out_path,std::ios::out);
        }

        fout<<tokens[2]<<" ";//type
        double truncated = std::stod(tokens[3]);
        truncated /=2 ;//(0,1,2) ->  0 (non-truncated) to 1 (truncated)
        fout<<truncated<<" ";

        size_t len=tokens.size();
        for(int i=4;i<len;++i){
            fout<<tokens[i]<<" ";
        }
        fout<<endl;

        curr_frame = frame;

        /**
    Kitti Object format:
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
        */
    }

    fout.close();
    fp.close();



    return 0;
}