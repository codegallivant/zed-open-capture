///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

// ----> Includes
#include <iostream>
#include <sstream>
#include <string>


#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <msgpack.hpp>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <array>

#include "videocapture.hpp"

// OpenCV includes
#include <opencv2/opencv.hpp>

#undef HAVE_OPENCV_VIZ // Uncomment if cannot use Viz3D for point cloud rendering

#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#include <opencv2/viz/viz3d.hpp>
#endif

// Sample includes
#include "calibration.hpp"
#include "stopwatch.hpp"
#include "stereo.hpp"
#include "ocv_display.hpp"

// <---- Includes
// #define USE_OCV_TAPI // Comment to use "normal" cv::Mat instead of CV::UMat
#define USE_HALF_SIZE_DISP // Comment to compute depth matching on full image frames
#define SOCKET_PUB

typedef struct CameraInfo {
    int height;
    int width;
    std::string distortion_model;
    std::vector<double> D;
    std::array<double, 9> K;
    std::array<double, 9> R;
    std::array<double, 12> P;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(CameraInfo, height, width, distortion_model, D, K, R, P);
} CameraInfo;

struct Image {
  std::vector<uchar> matrix;
  int rows = 0;
  int cols = 0;
  int type = 0;
  MSGPACK_DEFINE(matrix, rows, cols, type);
};

CameraInfo parse_camera_config(const std::string& filepath, const std::string& camera_section) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    CameraInfo camera_info;
    std::string line;
    std::string current_section;
    bool found_section = false;
    double fx = 0, fy = 0, cx = 0, cy = 0;

    // Set resolution based on camera section
    if (camera_section.find("2K") != std::string::npos) {
        camera_info.width = 2208;
        camera_info.height = 1242;
    } else if (camera_section.find("FHD") != std::string::npos) {
        camera_info.width = 1920;
        camera_info.height = 1080;
    } else if (camera_section.find("HD") != std::string::npos) {
        camera_info.width = 1280;
        camera_info.height = 720;
    } else if (camera_section.find("VGA") != std::string::npos) {
        camera_info.width = 672;
        camera_info.height = 376;
    }

    // Initialize matrices with zeros
    camera_info.D.resize(5, 0.0);  // k1, k2, p1, p2, k3
    std::fill(camera_info.K.begin(), camera_info.K.end(), 0.0);
    std::fill(camera_info.R.begin(), camera_info.R.end(), 0.0);
    std::fill(camera_info.P.begin(), camera_info.P.end(), 0.0);

    // Set R to identity matrix
    camera_info.R[0] = 1.0;
    camera_info.R[4] = 1.0;
    camera_info.R[8] = 1.0;

    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;

        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            if (current_section == camera_section) {
                found_section = true;
            } else if (found_section) {
                break;
            }
            continue;
        }

        if (!found_section) continue;

        size_t delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos) continue;

        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);
        double val = std::stod(value);

        if (key == "fx") {
            fx = val;
        } else if (key == "fy") {
            fy = val;
        } else if (key == "cx") {
            cx = val;
        } else if (key == "cy") {
            cy = val;
        } else if (key == "k1") {
            camera_info.D[0] = val;
        } else if (key == "k2") {
            camera_info.D[1] = val;
        } else if (key == "p1") {
            camera_info.D[2] = val;
        } else if (key == "p2") {
            camera_info.D[3] = val;
        } else if (key == "k3") {
            camera_info.D[4] = val;
        }
    }

    // Fill K matrix (camera intrinsics)
    camera_info.K[0] = fx;    // fx
    camera_info.K[2] = cx;    // cx
    camera_info.K[4] = fy;    // fy
    camera_info.K[5] = cy;    // cy
    camera_info.K[8] = 1.0;   // scale

    // Fill P matrix (projection matrix = K * [R|t])
    camera_info.P[0] = fx;    // fx
    camera_info.P[2] = cx;    // cx
    camera_info.P[5] = fy;    // fy
    camera_info.P[6] = cy;    // cy
    camera_info.P[10] = 1.0;  // scale

    camera_info.distortion_model = "plumb_bob";

    return camera_info;
}



int main(int argc, char *argv[])
{
    // ----> Silence unused warning
    (void)argc;
    (void)argv;
    // <---- Silence unused warning

// Socket
    zmq::context_t context(1);
    zmq::socket_t depth_socket(context, ZMQ_PUB);
    zmq::socket_t camera_info_socket(context, ZMQ_PUB);
    depth_socket.bind("tcp://*:"+std::to_string(5555));
    camera_info_socket.bind("tcp://*:"+std::to_string(5556));
// End Socket


    sl_oc::VERBOSITY verbose = sl_oc::VERBOSITY::INFO;

    // ----> Set Video parameters
    sl_oc::video::VideoParams params;
#ifdef EMBEDDED_ARM
    params.res = sl_oc::video::RESOLUTION::VGA;
#else
    params.res = sl_oc::video::RESOLUTION::HD720;
#endif
    params.fps = sl_oc::video::FPS::FPS_30;
    params.verbose = verbose;
    // <---- Set Video parameters

    // ----> Create Video Capture
    sl_oc::video::VideoCapture cap(params);
    if( !cap.initializeVideo(-1) )
    {
        std::cerr << "Cannot open camera video capture" << std::endl;
        std::cerr << "See verbosity level for more details." << std::endl;

        return EXIT_FAILURE;
    }
    int sn = cap.getSerialNumber();
    std::cout << "Connected to camera sn: " << sn << std::endl;
    // <---- Create Video Capture

    // ----> Retrieve calibration file from Stereolabs server
    std::string calibration_file;
    // ZED Calibration
    unsigned int serial_number = sn;
    // Download camera calibration file
    if( !sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file) )
    {
        std::cerr << "Could not load calibration file from Stereolabs servers" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Calibration file found. Loading..." << std::endl;

    // ----> Frame size
    int w,h;
    cap.getFrameSize(w,h);
    // <---- Frame size

    // ----> Initialize calibration
    cv::Mat map_left_x, map_left_y;
    cv::Mat map_right_x, map_right_y;
    cv::Mat cameraMatrix_left, cameraMatrix_right;
    double baseline=0;
    sl_oc::tools::initCalibration(calibration_file, cv::Size(w/2,h), map_left_x, map_left_y, map_right_x, map_right_y,
                                  cameraMatrix_left, cameraMatrix_right, &baseline);

    double fx = cameraMatrix_left.at<double>(0,0);
    double fy = cameraMatrix_left.at<double>(1,1);
    double cx = cameraMatrix_left.at<double>(0,2);
    double cy = cameraMatrix_left.at<double>(1,2);

    std::cout << " Camera Matrix L: \n" << cameraMatrix_left << std::endl << std::endl;
    std::cout << " Camera Matrix R: \n" << cameraMatrix_right << std::endl << std::endl;

#ifdef USE_OCV_TAPI
    cv::UMat map_left_x_gpu = map_left_x.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat map_left_y_gpu = map_left_y.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat map_right_x_gpu = map_right_x.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat map_right_y_gpu = map_right_y.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
#endif
    // ----> Initialize calibration

    // ----> Declare OpenCV images
#ifdef USE_OCV_TAPI
    cv::UMat frameYUV;  // Full frame side-by-side in YUV 4:2:2 format
    cv::UMat frameBGR(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Full frame side-by-side in BGR format
    cv::UMat left_raw(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left unrectified image
    cv::UMat right_raw(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right unrectified image
    cv::UMat left_rect(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left rectified image
    cv::UMat right_rect(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right rectified image
    cv::UMat left_for_matcher(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left image for the stereo matcher
    cv::UMat right_for_matcher(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right image for the stereo matcher
    cv::UMat left_disp_half(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Half sized disparity map
    cv::UMat left_disp(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Full output disparity
    cv::UMat left_disp_float(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Final disparity map in float32
    cv::UMat left_disp_image(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Normalized and color remapped disparity map to be displayed
    cv::UMat left_depth_map(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Depth map in float32
#else
    cv::Mat frameBGR, left_raw, left_rect, right_raw, right_rect, frameYUV, left_for_matcher, right_for_matcher, left_disp_half,left_disp,left_disp_float, left_disp_vis, left_disp_image, left_depth_map;
#endif
    // <---- Declare OpenCV images

    // ----> Stereo matcher initialization
    sl_oc::tools::StereoSgbmPar stereoPar;

    //Note: you can use the tool 'zed_open_capture_depth_tune_stereo' to tune the parameters and save them to YAML
    if(!stereoPar.load())
    {
        stereoPar.save(); // Save default parameters.
    }

    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(stereoPar.minDisparity,stereoPar.numDisparities,stereoPar.blockSize);
    left_matcher->setMinDisparity(stereoPar.minDisparity);
    left_matcher->setNumDisparities(stereoPar.numDisparities);
    left_matcher->setBlockSize(stereoPar.blockSize);
    left_matcher->setP1(stereoPar.P1);
    left_matcher->setP2(stereoPar.P2);
    left_matcher->setDisp12MaxDiff(stereoPar.disp12MaxDiff);
    left_matcher->setMode(stereoPar.mode);
    left_matcher->setPreFilterCap(stereoPar.preFilterCap);
    left_matcher->setUniquenessRatio(stereoPar.uniquenessRatio);
    left_matcher->setSpeckleWindowSize(stereoPar.speckleWindowSize);
    left_matcher->setSpeckleRange(stereoPar.speckleRange);

    stereoPar.print();
    // <---- Stereo matcher initialization


    // ----> Point Cloud
    cv::Mat cloudMat;

#ifdef HAVE_OPENCV_VIZ
    cv::viz::Viz3d pc_viewer = cv::viz::Viz3d( "Point Cloud" );
#endif
    // <---- Point Cloud


    uint64_t last_ts=0; // Used to check new frame arrival

    // Infinite video grabbing loop
    while (1)
    {
        // Get a new frame from camera
        const sl_oc::video::Frame frame = cap.getLastFrame();

        // ----> If the frame is valid we can convert, rectify and display it
        if(frame.data!=nullptr && frame.timestamp!=last_ts)
        {
            last_ts = frame.timestamp;

            // ----> Conversion from YUV 4:2:2 to BGR for visualization
#ifdef USE_OCV_TAPI
            cv::Mat frameYUV_cpu = cv::Mat( frame.height, frame.width, CV_8UC2, frame.data );
            frameYUV = frameYUV_cpu.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_HOST_MEMORY);
#else
            frameYUV = cv::Mat( frame.height, frame.width, CV_8UC2, frame.data );
#endif
            cv::cvtColor(frameYUV,frameBGR,cv::COLOR_YUV2BGR_YUYV);
            // <---- Conversion from YUV 4:2:2 to BGR for visualization

            // ----> Extract left and right images from side-by-side
            left_raw = frameBGR(cv::Rect(0, 0, frameBGR.cols / 2, frameBGR.rows));
            right_raw = frameBGR(cv::Rect(frameBGR.cols / 2, 0, frameBGR.cols / 2, frameBGR.rows));
            // <---- Extract left and right images from side-by-side

            // ----> Apply rectification
            sl_oc::tools::StopWatch remap_clock;
#ifdef USE_OCV_TAPI
            cv::remap(left_raw, left_rect, map_left_x_gpu, map_left_y_gpu, cv::INTER_AREA );
            cv::remap(right_raw, right_rect, map_right_x_gpu, map_right_y_gpu, cv::INTER_AREA );
#else
            cv::remap(left_raw, left_rect, map_left_x, map_left_y, cv::INTER_AREA );
            cv::remap(right_raw, right_rect, map_right_x, map_right_y, cv::INTER_AREA );
#endif
            double remap_elapsed = remap_clock.toc();
            std::stringstream remapElabInfo;
            remapElabInfo << "Rectif. processing: " << remap_elapsed << " sec - Freq: " << 1./remap_elapsed;
            // <---- Apply rectification

            // ----> Stereo matching
            sl_oc::tools::StopWatch stereo_clock;
            double resize_fact = 1.0;
#ifdef USE_HALF_SIZE_DISP
            resize_fact = 0.5;
            // Resize the original images to improve performances
            cv::resize(left_rect,  left_for_matcher,  cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
            cv::resize(right_rect, right_for_matcher, cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
#else
            left_for_matcher = left_rect; // No data copy
            right_for_matcher = right_rect; // No data copy
#endif
            // Apply stereo matching
            left_matcher->compute(left_for_matcher, right_for_matcher,left_disp_half);

            left_disp_half.convertTo(left_disp_float,CV_32FC1);
            cv::multiply(left_disp_float,1./16.,left_disp_float); // Last 4 bits of SGBM disparity are decimal

#ifdef USE_HALF_SIZE_DISP
            cv::multiply(left_disp_float,2.,left_disp_float); // Last 4 bits of SGBM disparity are decimal
            cv::Mat tmp = left_disp_float; // Required for OpenCV 3.2
            cv::resize(tmp, left_disp_float, cv::Size(), 1./resize_fact, 1./resize_fact, cv::INTER_AREA);
#else
            left_disp = left_disp_float;
#endif
            double elapsed = stereo_clock.toc();
            std::stringstream stereoElabInfo;
            stereoElabInfo << "Stereo processing: " << elapsed << " sec - Freq: " << 1./elapsed;
            // <---- Stereo matching

            // ----> Show frames
            sl_oc::tools::showImage("Right rect.", right_rect, params.res,true, remapElabInfo.str());
            sl_oc::tools::showImage("Left rect.", left_rect, params.res,true, remapElabInfo.str());
            // <---- Show frames

            // ----> Show disparity image
            cv::add(left_disp_float,-static_cast<double>(stereoPar.minDisparity-1),left_disp_float); // Minimum disparity offset correction
            cv::multiply(left_disp_float,1./stereoPar.numDisparities,left_disp_image,255., CV_8UC1 ); // Normalization and rescaling

            cv::applyColorMap(left_disp_image,left_disp_image,cv::COLORMAP_JET); // COLORMAP_INFERNO is better, but it's only available starting from OpenCV v4.1.0

            sl_oc::tools::showImage("Disparity", left_disp_image, params.res,true, stereoElabInfo.str());
            // <---- Show disparity image

            // ----> Extract Depth map
            // The DISPARITY MAP can be now transformed in DEPTH MAP using the formula
            // depth = (f * B) / disparity
            // where 'f' is the camera focal, 'B' is the camera baseline, 'disparity' is the pixel disparity

            double num = static_cast<double>(fx*baseline);
            cv::divide(num,left_disp_float,left_depth_map);

            float central_depth = left_depth_map.at<float>(left_depth_map.rows/2, left_depth_map.cols/2 );
            std::cout << "Depth of the central pixel: " << central_depth << " mm" << std::endl;
            // <---- Extract Depth map

            // ----> Create Point Cloud
            sl_oc::tools::StopWatch pc_clock;
            size_t buf_size = static_cast<size_t>(left_depth_map.cols * left_depth_map.rows);
            std::vector<cv::Vec3d> buffer( buf_size, cv::Vec3f::all( std::numeric_limits<float>::quiet_NaN() ) );
            cv::Mat depth_map_cpu = left_depth_map;
            float* depth_vec = (float*)(&(depth_map_cpu.data[0]));

#ifdef SOCKET_PUB
    cv::Mat img = depth_map_cpu;
    Image img_data;
    img_data.matrix = std::vector<uchar>(img.data, img.data + (img.rows * img.cols * img.channels()));
    img_data.rows = img.rows;
    img_data.cols = img.cols;
    img_data.type = img.type();

    /* packed/serialize the data using msgpack */
    msgpack::sbuffer serialized_img;
    msgpack::pack(&serialized_img, img_data);

    zmq::message_t packed_msg(serialized_img.size());
    std::memcpy(packed_msg.data(), serialized_img.data(), serialized_img.size());

    CameraInfo camera_info = parse_camera_config("/home/cdgr/zed/settings/SN33587609.conf", "RIGHT_CAM_HD");

    // Serialize CameraInfo
    nlohmann::json j = camera_info;
    std::string serialized = j.dump();

    zmq::message_t camera_info_msg(serialized.size());
    memcpy(camera_info_msg.data(), serialized.data(), serialized.size());

    depth_socket.send(packed_msg, zmq::send_flags::none);
    camera_info_socket.send(camera_info_msg, zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
#endif

#pragma omp parallel for
            for(size_t idx=0; idx<buf_size;idx++ )
            {
                size_t r = idx/left_depth_map.cols;
                size_t c = idx%left_depth_map.cols;
                double depth = static_cast<double>(depth_vec[idx]);
                //std::cout << depth << " ";
                if(!isinf(depth) && depth >=0 && depth > stereoPar.minDepth_mm && depth < stereoPar.maxDepth_mm)
                {
                    buffer[idx].val[2] = depth; // Z
                    buffer[idx].val[0] = (c-cx)*depth/fx; // X
                    buffer[idx].val[1] = (r-cy)*depth/fy; // Y
                }
            }

            cloudMat = cv::Mat( left_depth_map.rows, left_depth_map.cols, CV_64FC3, &buffer[0] ).clone();

            double pc_elapsed = stereo_clock.toc();
            std::stringstream pcElabInfo;
//            pcElabInfo << "Point cloud processing: " << pc_elapsed << " sec - Freq: " << 1./pc_elapsed;
            //std::cout << pcElabInfo.str() << std::endl;
            // <---- Create Point Cloud
        }


        // ----> Keyboard handling
        int key = cv::waitKey( 5 );
        if(key=='q' || key=='Q') // Quit
            break;
        // <---- Keyboard handling

#ifdef HAVE_OPENCV_VIZ
        // ----> Show Point Cloud
        cv::viz::WCloud cloudWidget( cloudMat, left_rect );
        cloudWidget.setRenderingProperty( cv::viz::POINT_SIZE, 1 );
        pc_viewer.showWidget( "Point Cloud", cloudWidget );
        pc_viewer.spinOnce(1);

        if(pc_viewer.wasStopped())
            break;
        // <---- Show Point Cloud
#endif
    }

    return EXIT_SUCCESS;
}


