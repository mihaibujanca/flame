#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <sophus/se3.hpp>
#include <stdio.h>
#include <sys/stat.h>
#include <cstdio>
#include <random>
#include <memory>
#include <limits>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <SLAMBenchAPI.h>
#include <io/SLAMFrame.h>
#include <io/sensor/CameraSensor.h>
#include <io/sensor/CameraSensorFinder.h>
#include <io/sensor/GroundTruthSensor.h>
#include <io/FrameSource.h>
#include <outputs/TrajectoryInterface.h>
#include <stdexcept>
#include <Eigen/Core>

#include "flame.h"
#include "utils/image_utils.h"
#include "utils/stats_tracker.h"
#include "utils/load_tracker.h"

#include "./utils.h"
#include "types.h"
#include <opencv2/highgui/highgui.hpp>
namespace fu = flame::utils;

static slambench::io::CameraSensor *grey_sensor = nullptr;
static slambench::io::CameraSensor *rgb_sensor;
static slambench::io::GroundTruthSensor *gt_sensor = nullptr;
static flame::Flame* sensor_;
cv::Mat1b *img_gray;
slambench::outputs::BaseOutput * gt_trajectory;
static cv::Mat* imRGB = NULL;
flame::Params params_;
static sb_uint2 inputSize;
int poseframe_subsample_factor; // Create a poseframe every this number of images.
double depth_scale_factor;
float max_angular_rate_;
static int img_id = 0;

Eigen::Matrix3f K;
Sophus::SE3f pose;
Sophus::SE3f prev_pose_;

double timestamp;
double prev_time_;
float min_depth;
static slambench::TimeStamp last_frame_timestamp;



//=========================================================================
// Variable parameters
//=========================================================================
double default_depth_scale_factor = 5000;
int default_poseframe_subsample_factor = 6;
float default_min_depth = 0.1;

int default_omp_num_threads = 4;
int default_omp_chunk_size = 1024;

float default_min_grad_mag = 5.0;
int default_detection_win_size = 16;
int default_win_size = 5;
int default_max_dropouts = 5;
float default_epipolar_line_var = 4.0;
bool default_do_nltgv2 = true;
bool default_adaptive_data_weights = false;
bool default_rescale_data = false;
bool default_init_with_prediction = true;
float default_idepth_var_max_graph = 0.01;
float default_data_factor = 0.15;
float default_step_x = 0.001;
float default_step_q = 125.0;
float default_theta = 0.25;
float default_min_height = -100000000000000.0;
float default_max_height = 100000000000000.0;
bool default_check_sticky_obstacles = false;
//=========================================================================
// SLAMBench output values
//=========================================================================

slambench::outputs::Output *frame_output = nullptr;
slambench::outputs::Output *pointcloud_output = nullptr;
static slambench::outputs::Output *pose_output = nullptr; // SLAMBench requires a pose output


bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings)
{

    slam_settings->addParameter(TypedParameter<double>("", "depth_scale_factor",
                                                       "Depth scaling factor to true distance",
                                                       &depth_scale_factor, &default_depth_scale_factor));

    slam_settings->addParameter(TypedParameter<int>("", "poseframe_subsample_factor",
                                                    "Number of pyramid levels used for features.",
                                                    &poseframe_subsample_factor, &default_poseframe_subsample_factor));
    slam_settings->addParameter(TypedParameter<float>("", "min_depth",
                                                      "Min depth.",
                                                      &min_depth, &default_min_depth));

    /*==================== Threading Params ====================*/

    slam_settings->addParameter(TypedParameter<int>("", "omp_num_threads",
                                                    "Number of threads used in parallel sections.",
                                                    &params_.omp_num_threads, &default_omp_num_threads));
    slam_settings->addParameter(TypedParameter<int>("", "omp_chunk_size",
                                                    "Number of items given to each thread",
                                                    &params_.omp_chunk_size, &default_omp_chunk_size));

    /*==================== Features Params ====================*/
    slam_settings->addParameter(TypedParameter<float>("", "min_grad_mag",
                                                      "Controls the minimum gradient magnitude for detected features..",
                                                      &params_.min_grad_mag, &default_min_grad_mag));
    params_.fparams.min_grad_mag = params_.min_grad_mag;
    slam_settings->addParameter(TypedParameter<int>("", "detection_win_size",
                                                    "// Features are detected on grid with cells of size (win_size x win_size).",
                                                    &params_.detection_win_size, &default_detection_win_size));

    slam_settings->addParameter(TypedParameter<int>("", "win_size",
                                                    "Window size for padding.",
                                                    &params_.zparams.win_size, &default_win_size));
    params_.fparams.win_size = params_.zparams.win_size;


    slam_settings->addParameter(TypedParameter<int>("", "max_dropouts",
                                                    "Maximum number of failed tracking attempts.",
                                                    &params_.max_dropouts, &default_max_dropouts));

    slam_settings->addParameter(TypedParameter<float>("", "epipolar_line_var",
                                                      "Epipolar line noise variance. Used for computing disparity variance.",
                                                      &params_.zparams.epipolar_line_var, &default_epipolar_line_var));

    /*==================== Regularizer Params ====================*/
    slam_settings->addParameter(TypedParameter<bool>("", "do_nltgv2",
                                                     "Apply planar regularization.",
                                                     &params_.do_nltgv2, &default_do_nltgv2));

    slam_settings->addParameter(TypedParameter<bool>("", "adaptive_data_weights",
                                                     "Set vertex data weights to inverse idepth variance.",
                                                     &params_.adaptive_data_weights, &default_adaptive_data_weights));

    slam_settings->addParameter(TypedParameter<bool>("", "rescale_data",
                                                     "Rescale data to have mean 1.",
                                                     &params_.rescale_data, &default_rescale_data));

    slam_settings->addParameter(TypedParameter<bool>("", "init_with_prediction",
                                                     "Initialize vertex idepths with predicted value from dense idepthmap.",
                                                     &params_.init_with_prediction, &default_init_with_prediction));

    slam_settings->addParameter(TypedParameter<float>("", "idepth_var_max",
                                                      "Maximum idepth var before feature can be added to graph.",
                                                      &params_.idepth_var_max_graph, &default_idepth_var_max_graph));

    slam_settings->addParameter(TypedParameter<float>("", "data_factor",
                                                      "Controls the balance between smoothing and data-fitting in the regularizer.",
                                                      &params_.rparams.data_factor, &default_data_factor));

    slam_settings->addParameter(TypedParameter<float>("", "step_x",
                                                      "Optimization primal step size.",
                                                      &params_.rparams.step_x, &default_step_x));

    slam_settings->addParameter(TypedParameter<float>("", "step_q",
                                                      "Optimization dual step size.",
                                                      &params_.rparams.step_q, &default_step_q));


    slam_settings->addParameter(TypedParameter<float>("", "theta",
                                                      "Extra-gradient step size.",
                                                      &params_.rparams.theta, &default_theta));


    slam_settings->addParameter(TypedParameter<float>("", "min_height",
                                                      "Minimum height of features that are added to graph.",
                                                      &params_.min_height, &default_min_height));

    slam_settings->addParameter(TypedParameter<float>("", "max_height",
                                                      "Maximum height of features that are added to graph.",
                                                      &params_.max_height, &default_max_height));

    slam_settings->addParameter(TypedParameter<bool>("", "check_sticky_obstacles",
                                                     "Check if idepths are being sucked towards the camera because of sticky obstacles.",
                                                     &params_.check_sticky_obstacles, &default_check_sticky_obstacles));

    std::cout << "FLAME configured" << std::endl;
    return true;
}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings)
{

    slambench::io::CameraSensorFinder sensor_finder;

    for(auto sensor : slam_settings->get_sensors())
        if(sensor->GetType() == slambench::io::GroundTruthSensor::kGroundTruthTrajectoryType)
        {
            gt_sensor = dynamic_cast<slambench::io::GroundTruthSensor*>(sensor);
        }

    if(gt_sensor == nullptr)
    {
        std::cerr << "Invalid sensors found, ground truth poses not found." << std::endl;
        return false;
    }

    grey_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "grey"}});
    if (grey_sensor == nullptr) {
        std::cerr << "Invalid sensors found, Grey not found." << std::endl;
        return false;
    }
    //// check sensor frame and pixel format
    if(grey_sensor->PixelFormat != slambench::io::pixelformat::G_I_8) {
        std::cerr << "Grey sensor is not in G_I_8 format" << std::endl;
        return false;
    }
    if(grey_sensor->FrameFormat != slambench::io::frameformat::Raster) {
        std::cerr << "Grey sensor is not in raster format" << std::endl;
    }

    rgb_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "rgb"}});
    //if (rgb_sensor == nullptr) {
    //    std::cerr << "Invalid sensors found, RGB not found." << std::endl;
    //    return false;
    //}

    K << grey_sensor->Intrinsics[0]*grey_sensor->Width, 0, grey_sensor->Intrinsics[1]*grey_sensor->Height,
            0, grey_sensor->Intrinsics[2]*grey_sensor->Width, grey_sensor->Intrinsics[3]*grey_sensor->Height,
            0,0,1;

    sensor_ = new flame::Flame(grey_sensor->Width,
                               grey_sensor->Height,
                               K,
                               K.inverse(),
                               params_);

    imRGB = new cv::Mat;// ( rgb_sensor->Height ,  rgb_sensor->Width, CV_8UC3);
    img_gray = new cv::Mat1b(grey_sensor->Height, grey_sensor->Width, CV_8UC1);
    inputSize = make_sb_uint2(grey_sensor->Width, grey_sensor->Height);


    //=========================================================================
    // DECLARE OUTPTUS
    //=========================================================================

    frame_output = new slambench::outputs::Output("Frame", slambench::values::VT_FRAME);
    frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(frame_output);
    frame_output->SetActive(true);


    rgb_frame_output = new slambench::outputs::Output("RGB Frame", slambench::values::VT_FRAME);
    //rgb_frame_output->SetKeepOnlyMostRecent(true);
    //slam_settings->GetOutputManager().RegisterOutput(rgb_frame_output);

    //pointcloud_output = new slambench::outputs::Output("PointCloud", slambench::values::VT_POINTCLOUD);
    //pointcloud_output->SetKeepOnlyMostRecent(true);
    //slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);
    //pointcloud_output->SetActive(true);

    pose_output = new slambench::outputs::Output("Pose", slambench::values::VT_POSE, true);
    slam_settings->GetOutputManager().RegisterOutput(pose_output);

    std::cout << "FLAME Initialization Successful." << std::endl;
    return true;

}

bool grey_ready = false, gt_ready = false;
std::vector<Sophus::SE3f> poses;
bool sb_update_frame(SLAMBenchLibraryHelper * slam_settings, slambench::io::SLAMFrame* s)
{
    if(s->FrameSensor == grey_sensor) {
        memcpy(img_gray->data, s->GetData(), s->GetSize());
        last_frame_timestamp = s->Timestamp;
        timestamp = s->Timestamp.ToS();
        s->FreeData();
        grey_ready = true;
        std::cerr<<"Gray READY"<<std::endl;
    }
    else if(s->FrameSensor == rgb_sensor) {
        memcpy(imRGB->data, s->GetData(), s->GetSize());
        last_frame_timestamp = s->Timestamp;
        timestamp = s->Timestamp.ToS();
        s->FreeData();
    }
    if( s->FrameSensor == gt_sensor)
    {
        Eigen::Matrix4f p;
        memcpy(p.data(), s->GetData(), s->GetSize());
        pose = Sophus::SE3f(p);
        s->FreeData();
        gt_ready = true;
    }

    return grey_ready && gt_ready;
}

bool sb_process_once (SLAMBenchLibraryHelper * slam_settings)  {
    std::cerr<<"Img id:"<<img_id<<std::endl;
    if (img_gray == NULL || img_gray->empty())
    {
        std::cout<<"PULAMEA"<<std::endl;
        return false;
    }
    img_id++;

    bool is_poseframe = (img_id % poseframe_subsample_factor) == 0;
    bool update_success = sensor_->update(timestamp, img_id, pose, *img_gray, is_poseframe);

    if (max_angular_rate_ > 0.0f) {
        // Check angle difference between last and current pose. If we're rotating,
        // we shouldn't publish output since it's probably too noisy.
        Eigen::Quaternionf q_delta = pose.unit_quaternion() *
                                     prev_pose_.unit_quaternion().inverse();
        float angle_delta = fu::fast_abs(Eigen::AngleAxisf(q_delta).angle());
        float angle_rate = angle_delta / (timestamp - prev_time_);

        prev_time_ = timestamp;
        prev_pose_ = pose;

    }
    grey_ready = false;
    gt_ready = false;
    return true;
}

bool sb_clean_slam_system()
{
    delete sensor_;
    delete gt_sensor;
    delete grey_sensor;

    return true;
}

bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *latest_output)
{
    slambench::TimeStamp ts = *latest_output;
    cv::Mat1f idepthmap;
    sensor_->getFilteredInverseDepthMap(&idepthmap);

    cv::Mat1f depth_est(idepthmap.rows, idepthmap.cols,
                        std::numeric_limits<float>::quiet_NaN());
    for (int ii = 0; ii < depth_est.rows; ++ii) {
        for (int jj = 0; jj < depth_est.cols; ++jj) {
            float idepth =  idepthmap(ii, jj);
            if (!std::isnan(idepth) && (idepth > 0)) {
                depth_est(ii, jj) = 1.0f/ idepth;
            }
        }
    }
    cv::Mat output(depth_est.rows,depth_est.cols,CV_8UC1);
    depth_est.convertTo(output,CV_8UC1);
    if(frame_output->IsActive()) {


        // add estimated depth
        frame_output->AddPoint(ts,
                               new slambench::values::FrameValue(inputSize.x, inputSize.y,
                                                                 slambench::io::pixelformat::G_I_8,
                                                                 (void*)output.data));
    }

    if(pose_output->IsActive()) {
        // Get the current pose as an eigen matrix
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        pose_output->AddPoint(ts, new slambench::values::PoseValue(pose.matrix()));
    }

    if(rgb_frame_output->IsActive()) {
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        rgb_frame_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGB_III_888, (void*)imRGB->data));
    }
    return true;
}


