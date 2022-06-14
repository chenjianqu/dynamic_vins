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


bool SplitMOT(int target_id,const string& source_file,const string& dst_file){

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

        out_file<<line<<endl;
    }
    fp.close();
    out_file.close();

    return true;
}



int main(int argc, char** argv)
{
    if(argc != 4){
        cerr<<"parameters number wrong!,usage: rosrun dynamic_vins_eval split_mot_to_tum"
              " ${object_id} ${mot_file} ${save_to_path}"<<endl;
        return 1;
    }

    int object_id = std::stoi(argv[1]);
    string mot_file = argv[2];
    string save_to_path = argv[3];

    SplitMOT( object_id,mot_file,save_to_path);

    return 0;
}
