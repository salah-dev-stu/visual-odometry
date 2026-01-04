/*
 * Visual Odometry - 2-Frame Relative Pose Estimation
 * Usage: ./vo_submission <image1> <image2> [-f focal_length] [-s scale]
 * Output: roll pitch yaw tx ty tz (degrees, unit vector)
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cmath>
#include <unordered_map>
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

double computeHomographyScore(const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2,
                               const cv::Mat& H, double threshold) {
    if (H.empty()) return 0;
    cv::Mat Hinv = H.inv();
    double score = 0;

    for (size_t i = 0; i < pts1.size(); i++) {
        cv::Mat p1 = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2 = (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, 1.0);

        cv::Mat p2_proj = H * p1;
        p2_proj /= p2_proj.at<double>(2);
        cv::Mat p1_proj = Hinv * p2;
        p1_proj /= p1_proj.at<double>(2);

        double err = std::pow(p2_proj.at<double>(0) - pts2[i].x, 2) +
                     std::pow(p2_proj.at<double>(1) - pts2[i].y, 2) +
                     std::pow(p1_proj.at<double>(0) - pts1[i].x, 2) +
                     std::pow(p1_proj.at<double>(1) - pts1[i].y, 2);

        if (err < threshold) score += (threshold - err);
    }
    return score;
}

double computeFundamentalScore(const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& pts2,
                                const cv::Mat& F, double threshold) {
    if (F.empty()) return 0;
    double score = 0;

    for (size_t i = 0; i < pts1.size(); i++) {
        cv::Mat p1 = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2 = (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, 1.0);

        cv::Mat Fp1 = F * p1;
        cv::Mat Ftp2 = F.t() * p2;
        double p2tFp1 = p2.dot(Fp1);

        double err = (p2tFp1 * p2tFp1) /
                     (Fp1.at<double>(0)*Fp1.at<double>(0) + Fp1.at<double>(1)*Fp1.at<double>(1) +
                      Ftp2.at<double>(0)*Ftp2.at<double>(0) + Ftp2.at<double>(1)*Ftp2.at<double>(1));

        if (err < threshold) score += (threshold - err);
    }
    return score;
}

bool selectBestHomographyPose(const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2,
                               const cv::Mat& K,
                               const std::vector<cv::Mat>& Rs,
                               const std::vector<cv::Mat>& ts,
                               cv::Mat& R_out, cv::Mat& t_out) {
    int bestIdx = -1;
    int maxInfront = 0;
    cv::Mat K_inv = K.inv();

    for (size_t i = 0; i < Rs.size(); i++) {
        int infront = 0;
        for (size_t j = 0; j < pts1.size(); j++) {
            cv::Mat p1 = K_inv * (cv::Mat_<double>(3,1) << pts1[j].x, pts1[j].y, 1.0);
            cv::Mat p2 = K_inv * (cv::Mat_<double>(3,1) << pts2[j].x, pts2[j].y, 1.0);
            cv::Mat p2_in_1 = Rs[i].t() * (p2 - ts[i]);
            if (p1.at<double>(2) > 0 && p2_in_1.at<double>(2) > 0) infront++;
        }
        if (infront > maxInfront) {
            maxInfront = infront;
            bestIdx = i;
        }
    }

    if (bestIdx < 0) return false;
    R_out = Rs[bestIdx].clone();
    t_out = ts[bestIdx].clone();
    return true;
}

void printZeroPose() {
    std::cout << "0.000000 0.000000 0.000000 0.000000 0.000000 1.000000" << std::endl;
}

int main(int argc, char** argv) {
    std::string img1_path, img2_path;
    double user_focal = -1;
    double scale = 0.5;  // Default: half resolution for speed

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            user_focal = std::stod(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            scale = std::stod(argv[++i]);
        } else if (img1_path.empty()) {
            img1_path = arg;
        } else if (img2_path.empty()) {
            img2_path = arg;
        }
    }

    if (img1_path.empty() || img2_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> [-f focal_length] [-s scale]" << std::endl;
        return 1;
    }

    cv::Mat img1_full = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2_full = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    if (img1_full.empty() || img2_full.empty()) {
        std::cerr << "Error: Could not load images" << std::endl;
        return 1;
    }

    // Downscale for speed
    cv::Mat img1, img2;
    if (scale < 1.0) {
        cv::resize(img1_full, img1, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(img2_full, img2, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        img1 = img1_full;
        img2 = img2_full;
    }

    // Adjust focal length for scale
    if (user_focal > 0) user_focal *= scale;

    int nfeatures = (scale >= 1.0) ? 2000 : (scale >= 0.5) ? 1000 : 500;
    auto orb = cv::ORB::create(nfeatures);
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
    } else {
        double best_score = -1;
        focal = img1.cols * 0.85;
        cv::Mat best_R, best_t;

        for (double mult = 0.5; mult <= 1.4; mult += 0.1) {
            double f = img1.cols * mult;
            cv::Mat K_test = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
            cv::Mat mask;
            cv::Mat E = cv::findEssentialMat(pts1, pts2, K_test, cv::USAC_MAGSAC, 0.999, 1.0, mask);

            if (E.empty() || E.rows != 3) continue;

            std::vector<cv::Point2f> pts1_in, pts2_in;
            for (size_t i = 0; i < pts1.size(); i++) {
                if (mask.at<uchar>(i)) {
                    pts1_in.push_back(pts1[i]);
                    pts2_in.push_back(pts2[i]);
                }
            }
            if (pts1_in.size() < 8) continue;

            cv::Mat R_test, t_test;
            int inliers = cv::recoverPose(E, pts1_in, pts2_in, K_test, R_test, t_test);
            if (inliers < 5) continue;

            cv::Mat K_inv = K_test.inv();
            double total_err = 0;
            int good = 0;
            for (size_t i = 0; i < pts1_in.size(); i++) {
                double err = computeSampsonError(pts1_in[i], pts2_in[i], E, K_inv);
                total_err += std::min(err, 1.0);
                if (err < 0.01) good++;
            }
            double score = (double)good / pts1_in.size() * 0.7 +
                           (1.0 / (1.0 + total_err / pts1_in.size() * 10.0)) * 0.3;

            if (score > best_score) {
                best_score = score;
                focal = f;
                best_R = R_test.clone();
                best_t = t_test.clone();
            }
        }

        if (best_score < 0 || best_R.empty()) {
            printZeroPose();
            return 0;
        }
        R = best_R;
        t = best_t;
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, cx, 0, focal, cy, 0, 0, 1);

    if (user_focal > 0) {
        cv::Mat H_mask, F_mask;
        cv::Mat H = cv::findHomography(pts1, pts2, cv::USAC_MAGSAC, 3.0, H_mask);
        cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, 1.0, 0.999, F_mask);

        if (F.empty()) {
            printZeroPose();
            return 0;
        }

        double scoreH = computeHomographyScore(pts1, pts2, H, 5.99 * 2);
        double scoreF = computeFundamentalScore(pts1, pts2, F, 3.84 * 2);
        double ratioH = scoreH / (scoreH + scoreF + 1e-10);

        if (ratioH > 0.45 && !H.empty()) {
            std::vector<cv::Mat> Rs, ts, normals;
            int solutions = cv::decomposeHomographyMat(H, K, Rs, ts, normals);

            if (solutions <= 0 || !selectBestHomographyPose(pts1, pts2, K, Rs, ts, R, t)) {
                cv::Mat E = K.t() * F * K;
                cv::Mat mask;
                cv::Mat E_refined = cv::findEssentialMat(pts1, pts2, K, cv::USAC_MAGSAC, 0.999, 1.0, mask);
                if (!E_refined.empty() && E_refined.rows == 3) E = E_refined;
                cv::recoverPose(E, pts1, pts2, K, R, t);
            }
        } else {
            cv::Mat mask;
            cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::USAC_MAGSAC, 0.999, 1.0, mask);

            if (E.empty() || E.rows != 3) E = K.t() * F * K;

            std::vector<cv::Point2f> pts1_in, pts2_in;
            for (size_t i = 0; i < pts1.size(); i++) {
                if (mask.empty() || mask.at<uchar>(i)) {
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
        }
    }

    if (R.empty() || t.empty()) {
        printZeroPose();
        return 0;
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
