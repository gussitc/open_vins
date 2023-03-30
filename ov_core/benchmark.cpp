#include <iostream>
#include <opencv2/opencv.hpp>
#include <gyro_aided_tracker.h>
#include <patch_match.h>
#include <chrono>

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
static const int circle_size = 2;

std::string file_path("/home/gustav/catkin_ws_ov/src/open_vins/ov_core/output/");
int write_to_file = false;
int pyr_levels = 3;
int half_patch_size = 10;
bool use_gyro = true;

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
    goodFeaturesToTrack(img0, p0, 10, 0.7, 30, Mat(), 7);
    p1 = p0;

    cv::Size win_size = Size(2 * half_patch_size + 1, 2 * half_patch_size + 1);
    std::vector<cv::Mat> imgpyr0, imgpyr1;
    cv::buildOpticalFlowPyramid(img0, imgpyr0, win_size, pyr_levels);
    cv::buildOpticalFlowPyramid(img1, imgpyr1, win_size, pyr_levels);

    // Convert keypoints into points (stupid opencv stuff)
    std::vector<cv::KeyPoint> kp0;
    for (size_t i = 0; i < p0.size(); i++) {
      kp0.push_back(KeyPoint(p0[i], 0));
    }

    imuCalib.Tbc = cv::Mat(4, 4, CV_32F, Tbc_data);
    GyroAidedTracker gyroPredictMatcher(img0, img1, kp0,
                      Rcl, imuCalib, Point3f(),
                      pCameraParams.mK, pCameraParams.mDistCoef, cv::Mat(),
                      GyroAidedTracker::IMAGE_ONLY_OPTICAL_FLOW_CONSIDER_ILLUMINATION,
                      GyroAidedTracker::PIXEL_AWARE_PREDICTION,
                      file_path, half_patch_size);

    // gyroPredictMatcher.TrackFeatures();
    gyroPredictMatcher.GyroPredictFeatures();

    PatchMatch patchMatch(&gyroPredictMatcher, half_patch_size, 30, pyr_levels,
                          use_gyro, false, true, true, false);

    auto start = chrono::high_resolution_clock::now();
    patchMatch.OpticalFlowMultiLevel();
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    printf("patch match runtime: %ld\n", duration.count());

    auto &gyro_predict = gyroPredictMatcher.mvPtGyroPredictUn;
    auto &predict_status = gyroPredictMatcher.mvStatus;
    auto &patch_match = gyroPredictMatcher.mvPtPredictAfterPatchMatchedUn;
    auto &predict_corners = gyroPredictMatcher.mvvPtPredictCornersUn;
    // auto patch_match = gyroPredictMatcher.mvPtPredictAfterPatchMatchedUn;

    // calculate optical flow
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

    // for (size_t i = 0; i < p1.size(); i++)
    // {
    //     if (!predict_status[i]) continue;
    //     p1[i] = gyro_predict[i];
    // }
    
    start = chrono::high_resolution_clock::now();
    calcOpticalFlowPyrLK(imgpyr0, imgpyr1, p0, p1, status, err, win_size, pyr_levels, criteria, OPTFLOW_USE_INITIAL_FLOW);
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    printf("opencv runtime: %ld\n", duration.count());

    vector<Point2f> good_new;
    Mat img0_out, img1_out;
    cvtColor(img0, img0_out, COLOR_GRAY2BGR);
    cvtColor(img1, img1_out, COLOR_GRAY2BGR);

    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            if (gyro_predict.size() > 0 && predict_status[i]){
                line(img0_out, p0[i], gyro_predict[i], COLOR_BLUE, 1, LINE_AA);
                circle(img1_out, gyro_predict[i], circle_size, COLOR_BLUE, -1, LINE_AA);

                cv::Point2f pt_tl = predict_corners[i][0];
                cv::Point2f pt_tr = predict_corners[i][1];
                cv::Point2f pt_bl = predict_corners[i][2];
                cv::Point2f pt_br = predict_corners[i][3];

                cv::line(img1_out, pt_tl, pt_tr, COLOR_BLUE, 1, cv::LINE_AA);
                cv::line(img1_out, pt_tr, pt_br, COLOR_BLUE, 1, cv::LINE_AA);
                cv::line(img1_out, pt_bl, pt_br, COLOR_BLUE, 1, cv::LINE_AA);
                cv::line(img1_out, pt_tl, pt_bl, COLOR_BLUE, 1, cv::LINE_AA);
            }
            line(img0_out, p0[i], p1[i], COLOR_GREEN, 1, LINE_AA);
            line(img1_out, p0[i], p1[i], COLOR_GREEN, 1, LINE_AA);
            line(img1_out, p0[i], patch_match[i], COLOR_YELLOW, 1, LINE_AA);
            circle(img0_out, p0[i], circle_size, COLOR_GREEN, -1, LINE_AA);
            circle(img1_out, p1[i], circle_size, COLOR_GREEN, -1, LINE_AA);
            circle(img1_out, patch_match[i], circle_size, COLOR_YELLOW, -1, LINE_AA);
            good_new.push_back(p1[i]);
        }
    }

    cout << Rcl << endl;
    Mat img_out;
    Range roi_x = Range(width/3,width*4/5);
    Range roi_y = Range(height/3 ,height);
    hconcat(img0_out(roi_y, roi_x), img1_out(roi_y, roi_x), img_out);
    // hconcat(img0_out, img1_out, img_out);
    resize(img_out, img_out, Size(), 2, 2);
    imshow("window", img_out);
    int key;
    do {
        key = waitKey(0);
        if (key == 'q') {
            break;
        }
        else if (key == 's'){
            printf("saving image\n");
            imwrite(file_path + "/img_out.png", img_out);
        }
    } while (key != 32 /*space key*/);
    return 0;
}