/*
 * Visual Odometry - 2-Frame Relative Pose Estimation
 * Usage: ./vo_submission <image1> <image2> [-f focal_length]
 * Output: roll pitch yaw tx ty tz (degrees, unit vector)
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <string>

double computeSampsonError(const cv::Point2f& pt1, const cv::Point2f& pt2,
                           const cv::Mat& E, const cv::Mat& K_inv) {
    cv::Mat p1 = (cv::Mat_<double>(3,1) << pt1.x, pt1.y, 1.0);
    cv::Mat p2 = (cv::Mat_<double>(3,1) << pt2.x, pt2.y, 1.0);
    cv::Mat p1_n = K_inv * p1;
    cv::Mat p2_n = K_inv * p2;

    cv::Mat Ep1 = E * p1_n;
    cv::Mat Etp2 = E.t() * p2_n;
    double p2tEp1 = p2_n.dot(Ep1);

    double denom = Ep1.at<double>(0) * Ep1.at<double>(0) +
                   Ep1.at<double>(1) * Ep1.at<double>(1) +
                   Etp2.at<double>(0) * Etp2.at<double>(0) +
                   Etp2.at<double>(1) * Etp2.at<double>(1);

    if (denom < 1e-10) return 1e10;

    return (p2tEp1 * p2tEp1) / denom;
}

double evaluateEssentialMatrix(const cv::Mat& E, const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2, const cv::Mat& K,
                               cv::Mat& R_out, cv::Mat& t_out) {
    cv::Mat R, t;
    cv::Mat mask;
    int inliers = cv::recoverPose(E, pts1, pts2, K, R, t, mask);

    if (inliers < 5) return 0.0;

    R_out = R.clone();
    t_out = t.clone();

    cv::Mat K_inv = K.inv();
    const double threshold = 0.01;
    int good_matches = 0;
    double total_error = 0.0;

    for (size_t i = 0; i < pts1.size(); i++) {
        double err = computeSampsonError(pts1[i], pts2[i], E, K_inv);
        total_error += std::min(err, 1.0);
        if (err < threshold) good_matches++;
    }

    double consistency_ratio = (double)good_matches / pts1.size();
    double mean_capped_error = total_error / pts1.size();
    double error_score = 1.0 / (1.0 + mean_capped_error * 10.0);

    return consistency_ratio * 0.7 + error_score * 0.3;
}

double estimateFocalLength(const std::vector<cv::Point2f>& pts1,
                          const std::vector<cv::Point2f>& pts2,
                          int img_width, int img_height,
                          cv::Mat& best_R, cv::Mat& best_t) {
    double cx = img_width / 2.0;
    double cy = img_height / 2.0;

    std::vector<double> focal_candidates;
    for (double mult = 0.4; mult <= 1.5; mult += 0.05)
        focal_candidates.push_back(img_width * mult);

    double best_score = -1;
    double best_focal = img_width * 0.85;

    for (double focal : focal_candidates) {
        cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, cx, 0, focal, cy, 0, 0, 1);
        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);

        if (E.empty() || E.rows != 3) continue;

        std::vector<cv::Point2f> pts1_in, pts2_in;
        for (size_t i = 0; i < pts1.size(); i++) {
            if (mask.at<uchar>(i)) {
                pts1_in.push_back(pts1[i]);
                pts2_in.push_back(pts2[i]);
            }
        }
        if (pts1_in.size() < 8) continue;

        cv::Mat R, t;
        double score = evaluateEssentialMatrix(E, pts1_in, pts2_in, K, R, t);

        if (score > best_score) {
            best_score = score;
            best_focal = focal;
            best_R = R.clone();
            best_t = t.clone();
        }
    }

    return best_focal;
}

void printZeroPose() {
    std::cout << "0.000000 0.000000 0.000000 0.000000 0.000000 1.000000" << std::endl;
}

int main(int argc, char** argv) {
    std::string img1_path, img2_path;
    double user_focal = -1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            user_focal = std::stod(argv[++i]);
        } else if (img1_path.empty()) {
            img1_path = arg;
        } else if (img2_path.empty()) {
            img2_path = arg;
        }
    }

    if (img1_path.empty() || img2_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> [-f focal_length]" << std::endl;
        std::cerr << "  -f focal_length: Optional focal length in pixels" << std::endl;
        return 1;
    }

    cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images" << std::endl;
        return 1;
    }

    auto orb = cv::ORB::create(3000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    if (kp1.size() < 30 || kp2.size() < 30) {
        printZeroPose();
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
        printZeroPose();
        return 0;
    }

    double cx = img1.cols / 2.0, cy = img1.rows / 2.0;
    double focal;
    cv::Mat R, t;

    if (user_focal > 0) {
        focal = user_focal;
        cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, cx, 0, focal, cy, 0, 0, 1);

        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);

        if (E.empty() || E.rows != 3) {
            printZeroPose();
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

        int inliers = cv::recoverPose(E, pts1_in, pts2_in, K, R, t);
        if (inliers < 5) {
            printZeroPose();
            return 0;
        }
    } else {
        focal = estimateFocalLength(pts1, pts2, img1.cols, img1.rows, R, t);

        if (R.empty() || t.empty()) {
            printZeroPose();
            return 0;
        }
    }

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
              << -t.at<double>(0) << " "
              << -t.at<double>(1) << " "
              << -t.at<double>(2) << std::endl;

    return 0;
}
