/*
 * Visual Odometry - 2-Frame Relative Pose Estimation
 * Usage: ./vo_submission <image1> <image2>
 * Output: roll pitch yaw tx ty tz (degrees, unit vector)
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cmath>
#include <unordered_map>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2>" << std::endl;
        return 1;
    }

    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images" << std::endl;
        return 1;
    }

    // Feature detection
    auto orb = cv::ORB::create(3000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    if (kp1.size() < 30 || kp2.size() < 30) {
        std::cout << "0.000000 0.000000 0.000000 0.000000 0.000000 1.000000" << std::endl;
        return 0;
    }

    // Symmetric matching
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
        std::cout << "0.000000 0.000000 0.000000 0.000000 0.000000 1.000000" << std::endl;
        return 0;
    }

    // Camera matrix (estimated)
    double cx = img1.cols / 2.0, cy = img1.rows / 2.0;
    double focal = img1.cols * 0.65;
    cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, cx, 0, focal, cy, 0, 0, 1);

    // Essential matrix
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);

    if (E.empty() || E.rows != 3) {
        std::cout << "0.000000 0.000000 0.000000 0.000000 0.000000 1.000000" << std::endl;
        return 0;
    }

    std::vector<cv::Point2f> pts1_in, pts2_in;
    for (size_t i = 0; i < pts1.size(); i++) {
        if (mask.at<uchar>(i)) {
            pts1_in.push_back(pts1[i]);
            pts2_in.push_back(pts2[i]);
        }
    }
    if (pts1_in.size() < 5) { pts1_in = pts1; pts2_in = pts2; }

    // Recover pose
    cv::Mat R, t;
    int inliers = cv::recoverPose(E, pts1_in, pts2_in, K, R, t);

    if (inliers < 5) {
        std::cout << "0.000000 0.000000 0.000000 0.000000 0.000000 1.000000" << std::endl;
        return 0;
    }

    // Euler angles (ZYX)
    double sy = std::sqrt(R.at<double>(0,0)*R.at<double>(0,0) + R.at<double>(1,0)*R.at<double>(1,0));
    double roll, pitch, yaw;
    if (sy > 1e-6) {
        roll  = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
        pitch = std::atan2(-R.at<double>(2,0), sy);
        yaw   = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        roll  = std::atan2(-R.at<double>(1,2), R.at<double>(1,1));
        pitch = std::atan2(-R.at<double>(2,0), sy);
        yaw   = 0;
    }

    std::cout << std::fixed << std::setprecision(6)
              << roll * 180.0 / CV_PI << " "
              << pitch * 180.0 / CV_PI << " "
              << yaw * 180.0 / CV_PI << " "
              << t.at<double>(0) << " "
              << t.at<double>(1) << " "
              << t.at<double>(2) << std::endl;

    return 0;
}
