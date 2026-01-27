/*
 * Visual Odometry - 2-Frame Relative Pose Estimation (ORB Frontend)
 * Usage: ./vo_submission <image1> <image2> [options]
 *   -f <focal>         Focal length (assumes fx=fy, cx=w/2, cy=h/2)
 *   -k <fx,fy,cx,cy>   Full camera intrinsics
 *   -s <scale>         Image scale factor (0.25, 0.5, 1.0)
 *   -m <file>          Output matched points to file
 *   -d                 Debug output
 *
 * Output format (4 lines):
 *   R11 R12 R13      (rotation matrix row 1)
 *   R21 R22 R23      (rotation matrix row 2)
 *   R31 R32 R33      (rotation matrix row 3)
 *   tx ty tz         (translation vector, unit length)
 *
 * Auto-focal estimation uses CVPR 2024 iterative method from PoseLib:
 * Kocur, Kyselica, Kukelova, "Robust Self-calibration of Focal Lengths from the Fundamental Matrix"
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <string>

#include "vo_geometry.hpp"

int main(int argc, char** argv) {
    std::string img1_path, img2_path, matches_file;
    double user_focal = -1;
    double user_fx = -1, user_fy = -1, user_cx = -1, user_cy = -1;
    double scale = -1;  // -1 means auto: full res for auto-focal, 0.5 for known focal
    bool debug = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            user_focal = std::stod(argv[++i]);
        } else if (arg == "-k" && i + 1 < argc) {
            std::string calib = argv[++i];
            std::replace(calib.begin(), calib.end(), ',', ' ');
            std::istringstream iss(calib);
            iss >> user_fx >> user_fy >> user_cx >> user_cy;
        } else if (arg == "-s" && i + 1 < argc) {
            scale = std::stod(argv[++i]);
        } else if (arg == "-m" && i + 1 < argc) {
            matches_file = argv[++i];
        } else if (arg == "-d") {
            debug = true;
        } else if (img1_path.empty()) {
            img1_path = arg;
        } else if (img2_path.empty()) {
            img2_path = arg;
        }
    }

    if (img1_path.empty() || img2_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  -f <focal>         Focal length in pixels (assumes fx=fy, cx=w/2, cy=h/2)" << std::endl;
        std::cerr << "  -k <fx,fy,cx,cy>   Full camera intrinsics" << std::endl;
        std::cerr << "  -s <scale>         Image scale factor (0.25, 0.5, 1.0)" << std::endl;
        std::cerr << "  -m <file>          Output matched points to file" << std::endl;
        std::cerr << "  -d                 Debug output" << std::endl;
        return 1;
    }

    bool has_full_calib = (user_fx > 0 && user_fy > 0 && user_cx > 0 && user_cy > 0);
    bool has_focal = (user_focal > 0);
    bool has_calib = has_full_calib || has_focal;

    cv::Mat img1_full = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2_full = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    if (img1_full.empty() || img2_full.empty()) {
        std::cerr << "Error: Could not load images" << std::endl;
        return 1;
    }

    if (scale < 0) {
        scale = has_calib ? 0.5 : 1.0;
    }

    cv::Mat img1, img2;
    if (scale < 1.0) {
        cv::resize(img1_full, img1, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(img2_full, img2, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        img1 = img1_full;
        img2 = img2_full;
    }

    if (user_focal > 0) user_focal *= scale;
    if (has_full_calib) {
        user_fx *= scale;
        user_fy *= scale;
        user_cx *= scale;
        user_cy *= scale;
    }

    int nfeatures = (scale >= 1.0) ? 2000 : (scale >= 0.5) ? 1000 : 500;
    auto orb = cv::ORB::create(nfeatures);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    if (kp1.size() < 30 || kp2.size() < 30) {
        vo_geometry::printZeroPose();
        return 0;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_12, knn_21;
    matcher.knnMatch(desc1, desc2, knn_12, 2);
    matcher.knnMatch(desc2, desc1, knn_21, 2);

    const float ratio = 0.75f;
    std::unordered_map<int, int> map_21;

    for (const auto& m : knn_21)
        if (m.size() >= 2 && m[0].distance < ratio * m[1].distance)
            map_21[m[0].trainIdx] = m[0].queryIdx;

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : knn_12) {
        if (m.size() >= 2 && m[0].distance < ratio * m[1].distance) {
            auto it = map_21.find(m[0].queryIdx);
            if (it != map_21.end() && it->second == m[0].trainIdx) {
                pts1.push_back(kp1[m[0].queryIdx].pt);
                pts2.push_back(kp2[m[0].trainIdx].pt);
            }
        }
    }

    if (pts1.size() < 10) {
        vo_geometry::printZeroPose();
        return 0;
    }

    // Output matches to file if requested (coordinates in original image space)
    if (!matches_file.empty()) {
        std::ofstream mf(matches_file);
        if (mf.is_open()) {
            mf << pts1.size() << std::endl;
            for (size_t i = 0; i < pts1.size(); i++) {
                mf << pts1[i].x / scale << " " << pts1[i].y / scale << " "
                   << pts2[i].x / scale << " " << pts2[i].y / scale << std::endl;
            }
            mf.close();
        }
    }

    // Build geometry config
    vo_geometry::GeometryConfig config;
    config.debug = debug;
    config.has_full_calib = has_full_calib;
    config.has_focal = has_focal;
    config.has_calib = has_calib;
    config.img_width = img1.cols;
    config.img_height = img1.rows;

    if (has_full_calib) {
        config.fx = user_fx;
        config.fy = user_fy;
        config.cx = user_cx;
        config.cy = user_cy;
    } else if (has_focal) {
        config.fx = config.fy = user_focal;
        config.cx = img1.cols / 2.0;
        config.cy = img1.rows / 2.0;
    } else {
        config.fx = config.fy = -1;
        config.cx = img1.cols / 2.0;
        config.cy = img1.rows / 2.0;
    }

    vo_geometry::PoseResult pose = vo_geometry::estimatePose(pts1, pts2, config);

    if (!pose.valid) {
        vo_geometry::printZeroPose();
        return 0;
    }

    vo_geometry::outputPose(pose.R, pose.t);
    return 0;
}
