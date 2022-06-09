#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <regex>
#include <map>

#include <Eigen/Dense>

using namespace std;



inline void split(const std::string& source, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
    std::regex re(delimiters);
    std::copy(std::sregex_token_iterator(source.begin(), source.end(),re,-1),
              std::sregex_token_iterator(),
              std::back_inserter(tokens));
}


bool SplitTrajectory(int target_id,const string& source_file,const string& dst_file,
                     std::unordered_map<string,vector<string>> &cam_pose){

    ///读取gt_file中的gt_id
    std::ifstream fp(source_file);
    if(!fp.is_open()){
        cerr<<"Can not open:"<<source_file<<endl;
        return false;
    }

    std::ofstream out_file(dst_file);
    if(!out_file.is_open()){
        cerr<<"Can not open:"<<dst_file<<endl;
        return false;
    }

    cout<<"target_id:"<<target_id<<endl;
    cout<<"source_file:"<<source_file<<endl;
    cout<<"dst_file:"<<dst_file<<endl;


    string line;
    while (getline(fp,line)){ //循环读取每行数据
        vector<string> tokens;
        split(line,tokens," ");
        //cout<<line<<endl;

        /**
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

        int frame = std::stoi(tokens[0]);
        int track_id = std::stoi(tokens[1]);

        if(track_id != target_id){
            continue;
        }

        double time = frame*0.05;
        //cout<<std::to_string(time)<<endl;

        ///获得位姿Two
        auto it_find = cam_pose.find(std::to_string(time));
        if(it_find == cam_pose.end()){
            cout<<"Can not find cam pose at time:"<<std::to_string(time)<<endl;
            continue;
        }
        auto &tokens_cam = it_find->second;
        Eigen::Vector3d t_wc(std::stod(tokens_cam[0]),std::stod(tokens_cam[1]),std::stod(tokens_cam[2]));
        Eigen::Quaterniond q_wc(std::stod(tokens_cam[6]),
                                std::stod(tokens_cam[3]),
                                std::stod(tokens_cam[4]),
                                std::stod(tokens_cam[5]));

        Eigen::Matrix3d R_wc = q_wc.matrix();

        ///构造位姿Tco
        Eigen::Vector3d t_co(std::stod(tokens[13]),std::stod(tokens[14]),std::stod(tokens[15]));
        //cout<<t_co.transpose()<<endl;

        double rotation_y = std::stod(tokens[16]);
        Eigen::Matrix3d R_oc;
        R_oc<<cos(rotation_y),0, -sin(rotation_y),   0,1,0,   sin(rotation_y),0,cos(rotation_y);
        Eigen::Matrix3d R_co = R_oc.transpose();

        Eigen::Matrix3d R_wo = R_wc * R_co;
        Eigen::Vector3d t_wo = R_wc * t_co + t_wc;

        Eigen::Quaterniond q(R_wo);
        /**
         * timestamp tx ty tz qx qy qz qw
         */
        out_file<<time<<" "<<
        t_wo.x()<<" "<<
        t_wo.y()<<" "<<
        t_wo.z()<<" "<<
        q.x()<<" "<<
        q.y()<<" "<<
        q.z()<<" "<<
        q.w()<<endl;
    }
    fp.close();
    out_file.close();

    return true;
}


std::unordered_map<string,vector<string>> ReadCameraPose(const string &pose_file)
{
    std::ifstream fp(pose_file);
    if(!fp.is_open()){
        cerr<<"Can not open:"<<pose_file<<endl;
        return {};
    }
    cout<<"pose_file:"<<pose_file<<endl;

    std::unordered_map<string,vector<string>> cam_pose;

    string line;
    while (getline(fp,line)){ //循环读取每行数据
        vector<string> tokens;
        split(line,tokens," ");
        //cout<<line<<endl;
        double time=std::stod(tokens[0]);
        cam_pose.insert({std::to_string(time),tokens});
    }

    return cam_pose;
}


int main(int argc, char** argv)
{
    if(argc != 7){
        cerr<<"parameters number wrong!,usage: rosrun dynamic_vins_eval convert_mot_to_trajectory"
              " ${gt_id} ${estimate_id} ${gt_file} ${estimate_file} ${gt_cam_pose_file} ${estimate_cam_pose_file}"<<endl;
        return 1;
    }

    int gt_id = std::stoi(argv[1]);
    int estimate_id = std::stoi(argv[2]);
    string gt_file = argv[3];
    string estimate_file = argv[4];

    string gt_cam_pose_file = argv[5];
    string estimate_cam_pose_file = argv[6];

    string save_ref_path=string(argv[1])+"_gt.txt";
    string save_estimate_path=string(argv[1])+"_estimate.txt";

    ///根据时间获取相机的位姿
    std::unordered_map<string,vector<string>> gt_cam_pose = ReadCameraPose(gt_cam_pose_file);
    std::unordered_map<string,vector<string>> estimate_cam_pose = ReadCameraPose(estimate_cam_pose_file);

    cout<<"ReadCameraPose finished"<<endl;
    cout<<"gt_cam_pose size:"<<gt_cam_pose.size()<<endl;
    cout<<"estimate_cam_pose size:"<<estimate_cam_pose.size()<<endl;

    SplitTrajectory( gt_id,gt_file,save_ref_path,gt_cam_pose);
    SplitTrajectory( estimate_id,estimate_file,save_estimate_path,estimate_cam_pose);

    return 0;
}
