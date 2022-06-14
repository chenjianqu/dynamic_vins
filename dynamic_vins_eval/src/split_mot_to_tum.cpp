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

    int cnt=0;

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
        Eigen::Vector3d t_wc(std::stod(tokens_cam[1]),std::stod(tokens_cam[2]),std::stod(tokens_cam[3]));
        Eigen::Quaterniond q_wc(std::stod(tokens_cam[7]),
                                std::stod(tokens_cam[4]),
                                std::stod(tokens_cam[5]),
                                std::stod(tokens_cam[6]));
        q_wc.normalize();
        Eigen::Matrix3d R_wc = q_wc.toRotationMatrix();

        ///构造位姿Tco
        Eigen::Vector3d t_co(std::stod(tokens[13]),std::stod(tokens[14]),std::stod(tokens[15]));
        //cout<<t_co.transpose()<<endl;

        ///9个yaw角(绕着y轴,因为y轴是垂直向下的)
        double yaw = std::stod(tokens[16]);
        //将yaw角限制到[-2pi,0]范围
        while(yaw>0) yaw -= (2*M_PI);
        while(yaw< (-2*M_PI)) yaw += (2*M_PI);
        //将yaw角限定在[-pi,0]上
        if(yaw < (-M_PI)){
            yaw += M_PI;
        }
        Eigen::Matrix3d R_oc;
        R_oc<<cos(yaw),0, -sin(yaw),   0,1,0,   sin(yaw),0,cos(yaw);
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

        cnt++;
        if(cnt==1){
       cout<<time<<" ";
            cout<<"t_co:"<<t_co.transpose()<<" ";
            cout<<"t_wc:"<<t_wc.transpose()<<" ";
            cout<<"t_wo:"<<t_wo.transpose()<<endl;
        }
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
    if(argc != 5){
        cerr<<"parameters number wrong!,usage: rosrun dynamic_vins_eval split_mot_to_tum"
              " ${gt_id} ${gt_file} ${gt_cam_pose_file} ${save_to_path}"<<endl;
        return 1;
    }

    int gt_id = std::stoi(argv[1]);
    string gt_file = argv[2];
    string gt_cam_pose_file = argv[3];
    string save_to_path = argv[4];

    ///根据时间获取相机的位姿
    std::unordered_map<string,vector<string>> gt_cam_pose = ReadCameraPose(gt_cam_pose_file);

    cout<<"ReadCameraPose finished"<<endl;
    cout<<"gt_cam_pose size:"<<gt_cam_pose.size()<<endl;

    SplitTrajectory( gt_id,gt_file,save_to_path,gt_cam_pose);

    return 0;
}
