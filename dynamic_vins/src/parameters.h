#ifndef DYNAMIC_VINS_PARAMETER_H
#define DYNAMIC_VINS_PARAMETER_H

#include <ros/ros.h>
#include <vector>
#include <fstream>
#include <map>
#include <iostream>
#include <exception>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/point_cloud.h>
#include <pcl/common/common.h>


#include <spdlog/spdlog.h>
#include "utility/utility.h"

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::pair;
using std::vector;

using namespace std::chrono_literals;
namespace fs=std::filesystem;


using PointT=pcl::PointXYZRGB;
using PointCloud=pcl::PointCloud<PointT>;

template <typename EigenType>
using EigenContainer = std::vector< EigenType ,Eigen::aligned_allocator<EigenType>>;

using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;
using Mat2d = Eigen::Matrix2d;
using Mat3d = Eigen::Matrix3d;
using Mat4d = Eigen::Matrix4d;
using Mat23d = Eigen::Matrix<double, 2, 3>;
using Mat24d = Eigen::Matrix<double, 2, 4>;
using Mat34d = Eigen::Matrix<double, 3, 4>;
using Mat35d = Eigen::Matrix<double, 3, 5>;
using Mat36d = Eigen::Matrix<double, 3, 6>;
using Mat37d = Eigen::Matrix<double, 3, 7>;
using Quatd = Eigen::Quaterniond;


using VecVector3d = EigenContainer<Eigen::Vector3d>;
using VecMatrix3d = EigenContainer<Eigen::Matrix3d>;


constexpr int INFER_IMAGE_LIST_SIZE=30;

constexpr double FOCAL_LENGTH = 460.0;
constexpr int WINDOW_SIZE = 10;
constexpr int NUM_OF_F = 1000;

constexpr int INSTANCE_FEATURE_SIZE=500;
constexpr int SIZE_SPEED=6;
constexpr int SIZE_BOX=3;


//图像归一化参数，注意是以RGB的顺序排序
inline float SOLO_IMG_MEAN[3]={123.675, 116.28, 103.53};
inline float SOLO_IMG_STD[3]={58.395, 57.12, 57.375};

constexpr int BATCH_SIZE=1;
constexpr int SOLO_TENSOR_CHANNEL=128;//张量的输出通道数应该是128

inline std::vector<float> SOLO_NUM_GRIDS={40, 36, 24, 16, 12};//各个层级划分的网格数
inline std::vector<float> SOLO_STRIDES={8, 8, 16, 32, 32};//各个层级的预测结果的stride


inline std::map<int,std::string> CocoLabelMap={
        {1, "person"}, {2, "bicycle"}, {3, "car"}, {4, "motorcycle"}, {5, "airplane"},
        {6, "bus"}, {7, "train"}, {8, "truck"}, {9, "boat"}, {10, "traffic light"},
        {11, "fire hydrant"}, {13, "stop sign"}, {14, "parking meter"}, {15, "bench"},
        {16, "bird"}, {17, "cat"}, {18, "dog"}, {19, "horse"}, {20, "sheep"}, {21, "cow"},
        {22, "elephant"}, {23, "bear"}, {24, "zebra"}, {25, "giraffe"}, {27, "backpack"},
        {28, "umbrella"}, {31, "handbag"}, {32, "tie"}, {33, "suitcase"}, {34, "frisbee"},
        {35, "skis"}, {36, "snowboard"}, {37, "sports ball"}, {38, "kite"}, {39, "baseball bat"},
        {40, "baseball glove"}, {41, "skateboard"}, {42, "surfboard"}, {43, "tennis racket"},
        {44, "bottle"}, {46, "wine glass"}, {47, "cup"}, {48, "fork"}, {49, "knife"}, {50, "spoon"},
        {51, "bowl"}, {52, "banana"}, {53, "apple"}, {54, "sandwich"}, {55, "orange"},
        {56, "broccoli"}, {57, "carrot"}, {58, "hot dog"}, {59, "pizza"}, {60, "donut"},
        {61, "cake"}, {62, "chair"}, {63, "couch"}, {64, "potted plant"}, {65, "bed"}, {67, "dining table"},
        {70, "toilet"}, {72, "tv"}, {73, "laptop"}, {74, "mouse"}, {75, "remote"}, {76, "keyboard"},
        {77, "cell phone"}, {78, "microwave"}, {79, "oven"}, {80, "toaster"},{ 81, "sink"},
        {82, "refrigerator"}, {84, "book"}, {85, "clock"},{ 86, "vase"}, {87, "scissors"},
        {88, "teddy bear"}, {89, "hair drier"}, {90, "toothbrush"}
};


inline std::shared_ptr<spdlog::logger> vioLogger;
inline std::shared_ptr<spdlog::logger> tkLogger;
inline std::shared_ptr<spdlog::logger> sgLogger;



inline std::vector<std::vector<int>> TENSOR_QUEUE_SHAPE{
        {1, 128, 12, 12},
        {1, 128, 16, 16},
        {1, 128, 24, 24},
        {1, 128, 36, 36},
        {1, 128, 40, 40},
        {1, 80, 12, 12},
        {1, 80, 16, 16},
        {1, 80, 24, 24},
        {1, 80, 36, 36},
        {1, 80, 40, 40},
        {1, 128, 96, 288}
};



enum SIZE_PARAMETERIZATION{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};


enum StateOrder{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

enum class SlamType{
    RAW,
    NAIVE,
    DYNAMIC
};

enum class DatasetType{
    VIODE,
    KITTI
};




enum class SolverFlag{
    INITIAL,
    NON_LINEAR
};

enum class MarginFlag{
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
};


class Config {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr=std::shared_ptr<Config>;

    explicit Config(const std::string &file_name);

    inline static double INIT_DEPTH;
    inline static double MIN_PARALLAX;
    inline static double ACC_N, ACC_W;
    inline static double GYR_N, GYR_W;

    inline static std::vector<Eigen::Matrix3d> RIC;
    inline static std::vector<Eigen::Vector3d> TIC;

    inline static Eigen::Vector3d G{0.0, 0.0, 9.8};

    inline static double BIAS_ACC_THRESHOLD,BIAS_GYR_THRESHOLD;
    inline static double SOLVER_TIME;
    inline static int NUM_ITERATIONS;
    inline static int ESTIMATE_EXTRINSIC;
    inline static int ESTIMATE_TD;
    inline static int ROLLING_SHUTTER;
    inline static std::string EX_CALIB_RESULT_PATH;
    inline static std::string VINS_RESULT_PATH;
    inline static std::string OUTPUT_FOLDER;
    inline static std::string IMU_TOPIC;
    inline static int ROW, COL;
    inline static double TD;
    inline static int NUM_OF_CAM;
    inline static int STEREO;
    inline static int USE_IMU;
    inline static std::map<int, Eigen::Vector3d> pts_gt;
    inline static std::string IMAGE0_TOPIC, IMAGE1_TOPIC,IMAGE0_SEGMENTATION_TOPIC,IMAGE1_SEGMENTATION_TOPIC;
    inline static std::string FISHEYE_MASK;
    inline static std::vector<std::string> CAM_NAMES;
    inline static int MAX_CNT; //每帧图像上的最多检测的特征数量
    inline static int MAX_DYNAMIC_CNT;
    inline static int MIN_DIST; //检测特征点时的最小距离
    inline static int MIN_DYNAMIC_DIST; //检测特征点时的最小距离
    inline static double F_THRESHOLD;
    inline static int SHOW_TRACK;
    inline static int FLOW_BACK; //是否反向计算光流，判断之前光流跟踪的特征点的质量

    inline static std::unordered_map<unsigned int,int> VIODE_Key2Index;
    inline static std::set<int> VIODE_DynamicIndex;

    inline static std::string DETECTOR_ONNX_PATH;
    inline static std::string DETECTOR_SERIALIZE_PATH;

    inline static int inputH,inputW,inputC;

    inline static SlamType SLAM;
    inline static DatasetType Dataset;
    inline static bool isInputSeg;

    inline static std::vector<std::string> CocoLabelVector;

    inline static std::string ESTIMATOR_LOG_PATH;
    inline static std::string ESTIMATOR_LOG_LEVEL;
    inline static std::string ESTIMATOR_LOG_FLUSH;
    inline static std::string FEATURE_TRACKER_LOG_PATH;
    inline static std::string FEATURE_TRACKER_LOG_LEVEL;
    inline static std::string FEATURE_TRACKER_LOG_FLUSH;
    inline static std::string SEGMENTOR_LOG_PATH;
    inline static std::string SEGMENTOR_LOG_LEVEL;
    inline static std::string SEGMENTOR_LOG_FLUSH;

    inline static int VISUAL_INST_DURATION;

    inline static std::string EXTRACTOR_MODEL_PATH;

    inline static int SOLO_NMS_PRE;
    inline static int SOLO_MAX_PER_IMG;
    inline static std::string SOLO_NMS_KERNEL;
    inline static float SOLO_NMS_SIGMA;
    inline static float SOLO_SCORE_THR;
    inline static float SOLO_MASK_THR;
    inline static float SOLO_UPDATE_THR;

    inline static int TRACKING_MAX_AGE;
    inline static int TRACKING_N_INIT;

    inline static std::atomic_bool ok{true};
};




#endif

