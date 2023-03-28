#include <iostream>
#include <opencv2/opencv.hpp>
#include <gyro_aided_tracker.h>

using namespace std;
using namespace cv;

double fx = 458.654;
double fy = 457.296;
double cx = 367.215;
double cy = 248.375;
double k1 = -0.28340811;
double k2 = 0.07395907;
double p1 = 0.00019359;
double p2 = 1.76187114e-05;
double k3 = 0;
int fps = 20;
int width = 752;
int height= 480;

int half_patch_size = 5;
std::string file_path("/home/gustav/catkin_ws_ov/src/open_vins/ov_core/output/");

size_t ref_num_keys = 0;
size_t num_new_keys = 0;
size_t num_good_tracks = 0;
CameraParams pCameraParams("cam_type", fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height, fps);
IMU::Calib imuCalib;

// FIXME(gustav): this is taken straight from the euroc imucam_chain.yaml
float Tbc_data[16] =   {0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
    -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
    0.0, 0.0, 0.0, 1.0};


static const cv::Scalar COLOR_BLUE(255, 0, 0);
static const cv::Scalar COLOR_GREEN(0, 255, 0);
static const cv::Scalar COLOR_RED(0, 0, 255);
static const cv::Scalar COLOR_WHITE(255, 255, 255);
static const cv::Scalar COLOR_YELLOW(0, 255, 255);
static const int circle_size = 3;

int pyr_levels = 3;
int write_to_file = false;

int main(){
    auto img0 = imread(file_path + "img0.png");
    auto img1 = imread(file_path + "img1.png");
    cvtColor(img0, img0, COLOR_BGR2GRAY);
    cvtColor(img1, img1, COLOR_BGR2GRAY);

    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(img0.size(), img0.type());

    Mat Rcl;
    FileStorage fp(file_path + "data.yml", FileStorage::READ);
    fp["Rcl"] >> Rcl;

    vector<Point2f> p0, p1;
    goodFeaturesToTrack(img0, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    p1 = p0;

    // Convert keypoints into points (stupid opencv stuff)
    std::vector<cv::KeyPoint> kp0;
    for (size_t i = 0; i < p0.size(); i++) {
      kp0.push_back(KeyPoint(p0[i], 0));
    }

    imuCalib.Tbc = cv::Mat(4, 4, CV_32F, Tbc_data);
    GyroAidedTracker gyroPredictMatcher(img0, img1, kp0,
                      Rcl, imuCalib, Point3f(),
                      pCameraParams.mK, pCameraParams.mDistCoef, cv::Mat(),
                      GyroAidedTracker::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION,
                      GyroAidedTracker::PIXEL_AWARE_PREDICTION,
                      file_path, half_patch_size);

    gyroPredictMatcher.GyroPredictFeatures();
    auto gyro_predict = gyroPredictMatcher.mvPtGyroPredictUn;
    auto predict_status = gyroPredictMatcher.mvStatus;

    // calculate optical flow
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(img0, img1, p0, p1, status, err, Size(15,15), 2, criteria, OPTFLOW_USE_INITIAL_FLOW);

    vector<Point2f> good_new;
    Mat img0_out, img1_out;
    cvtColor(img0, img0_out, COLOR_GRAY2BGR);
    cvtColor(img1, img1_out, COLOR_GRAY2BGR);

    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            if (predict_status[i]){
                line(img0_out, p0[i], gyro_predict[i], COLOR_WHITE, 1, LINE_AA);
            }
            good_new.push_back(p1[i]);
            line(img0_out, p0[i], p1[i], COLOR_YELLOW, 1, LINE_AA);
            line(img1_out, p0[i], p1[i], COLOR_YELLOW, 1, LINE_AA);
            circle(img0_out, p0[i], circle_size, COLOR_BLUE, -1);
            // circle(img1_out, p0[i], circle_size, COLOR_BLUE, -1);
            // circle(img0_out, p1[i], circle_size, COLOR_GREEN, -1);
            circle(img1_out, p1[i], circle_size, COLOR_GREEN, -1);
        }
    }

    cout << Rcl << endl;
    Mat img_out;
    hconcat(img0_out, img1_out, img_out);
    imshow("img", img_out);
    waitKey(0);
    return 0;
}