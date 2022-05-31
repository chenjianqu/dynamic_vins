
#include "oxts_parser.h"

#include <fstream>
#include <iostream>

using namespace std;


void SaveOxtsTrajectory(const string &data_path,const string &save_path){
    ofstream fout(save_path,std::ios::out);

    vector<Eigen::Matrix4d> pose ;
    ParseOxts(pose,data_path);

    double time=0;

    for(auto &T: pose){
        Eigen::Matrix3d R=T.topLeftCorner(3,3);
        Eigen::Quaterniond q(R);
        Eigen::Vector3d t=T.block<3,1>(0,3);
        fout<<time<<" "<<t.x()<<" "<<t.y()<<" "<<t.z()<<" "<<q.x()<<" "
        <<q.y()<<" "<<q.z()<<" "<<q.w()<<endl;

        time+=0.05;
    }


    fout.close();

    cout<<"Save number "<<pose.size()<<endl;

}



int main(int argc, char** argv)
{
    if(argc!=3 && argc!=1){
        cerr<<"usage:./save_oxts [oxts_source_path] [oxts_target_path]"<<endl;
        return -1;
    }

    string data_path = "/home/chen/CLionProjects/CV_Tools/cv_ws/src/kitti_pub/data/oxts/0007.txt";
    string save_path = "save_trajectory.txt";

    if(argc==3){
        data_path=argv[1];
        save_path=argv[2];
    }

    SaveOxtsTrajectory(data_path,save_path);

    return 0;
}