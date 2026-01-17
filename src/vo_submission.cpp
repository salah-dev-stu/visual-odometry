/*
 * Visual Odometry - 2-Frame Relative Pose Estimation
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

// PoseLib includes for CVPR 2024 focal estimation
#include <PoseLib/misc/decompositions.h>
#include <PoseLib/misc/colmap_models.h>
#include <Eigen/Dense>

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

// Compute median parallax angle from triangulated points
// Returns parallax in degrees - low parallax indicates pure rotation or small baseline
double computeMedianParallax(const cv::Mat& R, const cv::Mat& t,
                              const std::vector<cv::Point2f>& pts1,
                              const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& K) {
    cv::Mat K_inv = K.inv();
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P2(3, 4, CV_64F);
    R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    t.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    // Camera centers
    cv::Mat O1 = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat O2 = -R.t() * t;

    std::vector<double> parallaxes;

    for (size_t i = 0; i < pts1.size(); i++) {
        cv::Mat p1_h = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2_h = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat p1_n = K_inv * p1_h;
        cv::Mat p2_n = K_inv * p2_h;

        // Triangulate
        cv::Mat A(4, 4, CV_64F);
        A.row(0) = p1_n.at<double>(0) * P1.row(2) - P1.row(0);
        A.row(1) = p1_n.at<double>(1) * P1.row(2) - P1.row(1);
        A.row(2) = p2_n.at<double>(0) * P2.row(2) - P2.row(0);
        A.row(3) = p2_n.at<double>(1) * P2.row(2) - P2.row(1);

        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        cv::Mat X = vt.row(3).t();
        if (std::abs(X.at<double>(3)) < 1e-10) continue;
        X = X / X.at<double>(3);
        cv::Mat X3D = X.rowRange(0, 3);

        // Skip points behind cameras
        if (X3D.at<double>(2) <= 0) continue;
        cv::Mat X3D_c2 = R * X3D + t;
        if (X3D_c2.at<double>(2) <= 0) continue;

        // Compute parallax angle (angle between rays from both cameras to 3D point)
        cv::Mat ray1 = X3D - O1;
        cv::Mat ray2 = X3D - O2;
        double norm1 = cv::norm(ray1);
        double norm2 = cv::norm(ray2);
        if (norm1 < 1e-10 || norm2 < 1e-10) continue;

        double cosParallax = ray1.dot(ray2) / (norm1 * norm2);
        cosParallax = std::min(1.0, std::max(-1.0, cosParallax));
        double parallaxDeg = std::acos(cosParallax) * 180.0 / CV_PI;
        parallaxes.push_back(parallaxDeg);
    }

    if (parallaxes.empty()) return 0.0;
    std::sort(parallaxes.begin(), parallaxes.end());
    return parallaxes[parallaxes.size() / 2];
}

// Check if Homography is "rotation-like" (H ≈ K*R*K⁻¹ for pure rotation)
// Returns orthogonality error - lower means more rotation-like
double checkHomographyRotationLike(const cv::Mat& H, const cv::Mat& K) {
    if (H.empty()) return 1e10;

    cv::Mat K_inv = K.inv();
    cv::Mat R_approx = K_inv * H * K;

    // Normalize by determinant to remove scale
    double det = cv::determinant(R_approx);
    if (std::abs(det) < 1e-10) return 1e10;
    if (det < 0) det = -det;
    double scale = std::pow(det, 1.0/3.0);
    R_approx = R_approx / scale;

    // Check how close to orthogonal: ||R^T * R - I||
    cv::Mat RtR = R_approx.t() * R_approx;
    return cv::norm(RtR - cv::Mat::eye(3, 3, CV_64F));
}

// Extract rotation from Homography for pure rotation case
// For pure rotation: H = K * R * K^(-1), so R = K^(-1) * H * K
bool extractRotationFromHomography(const cv::Mat& H, const cv::Mat& K, cv::Mat& R_out) {
    if (H.empty()) return false;

    cv::Mat K_inv = K.inv();
    cv::Mat R_approx = K_inv * H * K;

    // Check if R_approx is close to a valid rotation matrix
    // A valid rotation has det(R) = 1 and R^T * R = I
    double det = cv::determinant(R_approx);
    if (det < 0) {
        R_approx = -R_approx;
        det = -det;
    }

    // Normalize to ensure det = 1 (handle scale in H)
    double scale = std::pow(det, 1.0/3.0);
    R_approx = R_approx / scale;

    // Force orthogonality using SVD: R = U * V^T
    cv::Mat w, u, vt;
    cv::SVD::compute(R_approx, w, u, vt);
    R_out = u * vt;

    // Verify it's a proper rotation (det = 1, not -1)
    if (cv::determinant(R_out) < 0) {
        R_out = -R_out;
    }

    // Check orthogonality error
    cv::Mat I = R_out * R_out.t();
    double ortho_err = cv::norm(I - cv::Mat::eye(3, 3, CV_64F));
    return ortho_err < 0.1;  // Allow small error
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
    // Output identity rotation matrix and zero translation
    std::cout << std::fixed << std::setprecision(6);
    // Rotation matrix (3x3 identity)
    std::cout << "1.000000 0.000000 0.000000" << std::endl;
    std::cout << "0.000000 1.000000 0.000000" << std::endl;
    std::cout << "0.000000 0.000000 1.000000" << std::endl;
    // Translation vector
    std::cout << "0.000000 0.000000 0.000000" << std::endl;
}

// Validate pose by triangulating points and checking quality (inspired by ORB-SLAM3 CheckRT)
// Returns number of points with positive depth in both cameras and low reprojection error
// Also computes depth statistics to help distinguish 3D vs planar scenes
int validatePose(const cv::Mat& R, const cv::Mat& t,
                 const std::vector<cv::Point2f>& pts1,
                 const std::vector<cv::Point2f>& pts2,
                 const cv::Mat& K, double reproj_thresh = 4.0,
                 double* depth_variance = nullptr, double* median_depth = nullptr) {
    cv::Mat K_inv = K.inv();
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);  // [I|0]
    cv::Mat P2(3, 4, CV_64F);
    R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    t.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    int nGood = 0;
    std::vector<double> depths;

    for (size_t i = 0; i < pts1.size(); i++) {
        // Normalize points
        cv::Mat p1_h = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2_h = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat p1_n = K_inv * p1_h;
        cv::Mat p2_n = K_inv * p2_h;

        // Triangulate using DLT
        cv::Mat A(4, 4, CV_64F);
        A.row(0) = p1_n.at<double>(0) * P1.row(2) - P1.row(0);
        A.row(1) = p1_n.at<double>(1) * P1.row(2) - P1.row(1);
        A.row(2) = p2_n.at<double>(0) * P2.row(2) - P2.row(0);
        A.row(3) = p2_n.at<double>(1) * P2.row(2) - P2.row(1);

        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        cv::Mat X = vt.row(3).t();
        if (std::abs(X.at<double>(3)) < 1e-10) continue;
        X = X / X.at<double>(3);
        cv::Mat X3D = X.rowRange(0, 3);

        // Check depth in camera 1 (must be positive)
        double z1 = X3D.at<double>(2);
        if (z1 <= 0) continue;

        // Check depth in camera 2 (must be positive)
        cv::Mat X3D_c2 = R * X3D + t;
        double z2 = X3D_c2.at<double>(2);
        if (z2 <= 0) continue;

        // Check reprojection error in camera 1
        double invZ1 = 1.0 / z1;
        double u1 = K.at<double>(0, 0) * X3D.at<double>(0) * invZ1 + K.at<double>(0, 2);
        double v1 = K.at<double>(1, 1) * X3D.at<double>(1) * invZ1 + K.at<double>(1, 2);
        double err1 = (u1 - pts1[i].x) * (u1 - pts1[i].x) + (v1 - pts1[i].y) * (v1 - pts1[i].y);
        if (err1 > reproj_thresh) continue;

        // Check reprojection error in camera 2
        double invZ2 = 1.0 / z2;
        double u2 = K.at<double>(0, 0) * X3D_c2.at<double>(0) * invZ2 + K.at<double>(0, 2);
        double v2 = K.at<double>(1, 1) * X3D_c2.at<double>(1) * invZ2 + K.at<double>(1, 2);
        double err2 = (u2 - pts2[i].x) * (u2 - pts2[i].x) + (v2 - pts2[i].y) * (v2 - pts2[i].y);
        if (err2 > reproj_thresh) continue;

        nGood++;
        depths.push_back(z1);
    }

    // Compute depth statistics for scene type detection
    if (depth_variance && median_depth && !depths.empty()) {
        std::sort(depths.begin(), depths.end());
        *median_depth = depths[depths.size() / 2];

        // Compute variance using median absolute deviation (more robust)
        double sum_sq_diff = 0;
        for (double d : depths) {
            double diff = d - *median_depth;
            sum_sq_diff += diff * diff;
        }
        *depth_variance = sum_sq_diff / depths.size();
    }

    return nGood;
}

// CVPR 2024 iterative focal length estimation using PoseLib
double estimateFocalPoseLib(const std::vector<cv::Point2f>& pts1,
                             const std::vector<cv::Point2f>& pts2,
                             double cx, double cy, double initial_focal,
                             bool debug = false) {
    // Compute Fundamental matrix
    cv::Mat F_cv = cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, 1.0, 0.999);
    if (F_cv.empty() || F_cv.rows != 3) {
        if (debug) std::cerr << "F matrix estimation failed" << std::endl;
        return initial_focal;  // Fallback to initial guess
    }

    // Convert cv::Mat to Eigen::Matrix3d
    Eigen::Matrix3d F;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            F(i, j) = F_cv.at<double>(i, j);
        }
    }

    // Create camera priors with initial focal estimate and principal point
    poselib::Camera camera1_prior(poselib::SimplePinholeCameraModel::model_id,
                                   std::vector<double>{initial_focal, cx, cy}, -1, -1);
    poselib::Camera camera2_prior(poselib::SimplePinholeCameraModel::model_id,
                                   std::vector<double>{initial_focal, cx, cy}, -1, -1);

    // Run iterative focal estimation (max 50 iterations)
    auto [camera1_out, camera2_out, iters] =
        poselib::focals_from_fundamental_iterative(F, camera1_prior, camera2_prior, 50);

    double f1 = camera1_out.focal();
    double f2 = camera2_out.focal();

    if (debug) {
        std::cerr << "PoseLib focal: f1=" << f1 << " f2=" << f2
                  << " iters=" << iters << " init=" << initial_focal << std::endl;
    }

    // Check for valid results (positive and reasonable focal lengths)
    // More lenient bounds: focal should be > 0.1 * image_width and < 3 * image_width
    double min_focal = initial_focal * 0.2;
    double max_focal = initial_focal * 3.0;
    if (f1 > min_focal && f2 > min_focal && f1 < max_focal && f2 < max_focal) {
        // Use average of the two estimated focal lengths for same camera
        return (f1 + f2) / 2.0;
    }

    return initial_focal;  // Fallback
}

int main(int argc, char** argv) {
    std::string img1_path, img2_path, matches_file;
    double user_focal = -1;
    double user_fx = -1, user_fy = -1, user_cx = -1, user_cy = -1;  // Full calibration
    double scale = -1;  // -1 means auto: full res for auto-focal, 0.5 for known focal
    bool debug = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            user_focal = std::stod(argv[++i]);
        } else if (arg == "-k" && i + 1 < argc) {
            // Parse full calibration: fx,fy,cx,cy
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

    // Check for valid calibration input
    bool has_full_calib = (user_fx > 0 && user_fy > 0 && user_cx > 0 && user_cy > 0);
    bool has_focal = (user_focal > 0);
    bool has_calib = has_full_calib || has_focal;

    cv::Mat img1_full = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2_full = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    if (img1_full.empty() || img2_full.empty()) {
        std::cerr << "Error: Could not load images" << std::endl;
        return 1;
    }

    // Auto-select scale: full res for auto-focal (needs more features), half for known focal
    if (scale < 0) {
        scale = has_calib ? 0.5 : 1.0;
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

    // Adjust calibration for scale
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

    // Set up camera intrinsics
    double fx, fy, cx, cy;
    if (has_full_calib) {
        fx = user_fx;
        fy = user_fy;
        cx = user_cx;
        cy = user_cy;
    } else if (has_focal) {
        fx = fy = user_focal;
        cx = img1.cols / 2.0;
        cy = img1.rows / 2.0;
    } else {
        // Will be set by auto-focal estimation
        fx = fy = -1;
        cx = img1.cols / 2.0;
        cy = img1.rows / 2.0;
    }

    cv::Mat R, t;

    if (has_calib) {
        // Use provided calibration - processing happens below
    } else {
        // Use CVPR 2024 iterative method from PoseLib for focal estimation
        double initial_focal = img1.cols * 0.85;  // Initial guess: typical webcam FOV
        fx = fy = estimateFocalPoseLib(pts1, pts2, cx, cy, initial_focal, debug);

        // Estimate pose with the computed focal length
        cv::Mat K_auto = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K_auto, cv::USAC_MAGSAC, 0.999, 1.0, mask);

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

        int inliers = cv::recoverPose(E, pts1_in, pts2_in, K_auto, R, t);
        if (debug) {
            std::cerr << "Auto-focal: inliers=" << inliers << " pts=" << pts1_in.size() << std::endl;
        }
        if (inliers < 5) {
            printZeroPose();
            return 0;
        }

        // Validate pose quality (reject if too few points pass triangulation checks)
        int nGood = validatePose(R, t, pts1_in, pts2_in, K_auto);
        double goodRatio = (double)nGood / pts1_in.size();
        if (debug) {
            std::cerr << "Auto-focal validation: nGood=" << nGood << "/" << pts1_in.size()
                      << " ratio=" << goodRatio << std::endl;
        }
        if (goodRatio < 0.7 || nGood < 8) {
            printZeroPose();
            return 0;
        }
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    if (has_calib) {
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

        if (debug) {
            std::cerr << "H/F: scoreH=" << scoreH << " scoreF=" << scoreF
                      << " ratioH=" << ratioH << std::endl;
        }

        // Try both E and H paths, pick the one with better validation
        cv::Mat R_E, t_E, R_H, t_H;
        double goodRatio_E = 0, goodRatio_H = 0;
        int nGood_E = 0, nGood_H = 0;
        bool E_valid = false, H_valid = false;

        // Always try E matrix path
        double depth_variance_E = 0.0, median_depth_E = 0.0;
        {
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

            int inliers = cv::recoverPose(E, pts1_in, pts2_in, K, R_E, t_E);
            if (inliers >= 5) {
                nGood_E = validatePose(R_E, t_E, pts1_in, pts2_in, K, 4.0, &depth_variance_E, &median_depth_E);
                goodRatio_E = (double)nGood_E / pts1_in.size();
                E_valid = (goodRatio_E >= 0.3 && nGood_E >= 5);  // Relaxed for comparison
            }
            if (debug) {
                std::cerr << "E path: inliers=" << inliers << " nGood=" << nGood_E
                          << " ratio=" << goodRatio_E << " valid=" << E_valid
                          << " depth_var=" << depth_variance_E << " median_depth=" << median_depth_E << std::endl;
            }
        }

        // Try H matrix path if H looks reasonable
        bool H_decomp_ok = false;
        if (!H.empty() && ratioH > 0.40) {
            std::vector<cv::Mat> Rs, ts, normals;
            int solutions = cv::decomposeHomographyMat(H, K, Rs, ts, normals);

            if (solutions > 0 && selectBestHomographyPose(pts1, pts2, K, Rs, ts, R_H, t_H)) {
                H_decomp_ok = true;  // Decomposition succeeded
                nGood_H = validatePose(R_H, t_H, pts1, pts2, K);
                goodRatio_H = (double)nGood_H / pts1.size();
                H_valid = (goodRatio_H >= 0.25 && nGood_H >= 5);
            }
            if (debug) {
                std::cerr << "H path: decomp=" << H_decomp_ok << " nGood=" << nGood_H
                          << " ratio=" << goodRatio_H << " valid=" << H_valid << std::endl;
            }
        }

        // Smart H/E selection combining:
        // 1. H/F score (ratioH) - scene geometry indicator
        // 2. Depth variance - distinguishes 3D vs planar scenes
        // 3. Direction agreement - consistency check between E and H
        bool useH = false;

        // Compare translation directions between E and H when both available
        double direction_agreement = 0.0;  // cos(angle) between t_E and t_H
        if (E_valid && H_decomp_ok && !t_E.empty() && !t_H.empty()) {
            cv::Mat t_E_norm = t_E / cv::norm(t_E);
            cv::Mat t_H_norm = t_H / cv::norm(t_H);
            direction_agreement = t_E_norm.dot(t_H_norm);
            if (debug) std::cerr << "E-H direction agreement: " << direction_agreement
                                 << " (angle: " << std::acos(std::abs(direction_agreement)) * 180.0 / CV_PI << "°)" << std::endl;
        }

        // Compute normalized depth variance (coefficient of variation)
        // High CV indicates 3D scene with varied depths; low CV indicates planar/distant scene
        double depth_cv = 0.0;
        if (median_depth_E > 0.01) {
            depth_cv = std::sqrt(depth_variance_E) / median_depth_E;
        }
        if (debug) std::cerr << "Depth CV (std/median): " << depth_cv << std::endl;

        // === PURE ROTATION DETECTION ===
        // Detect pure rotation: H explains motion well + very low parallax + E gives unreliable translation
        // When detected, extract rotation from H and set translation to zero
        double median_parallax = 0.0;
        bool pure_rotation_detected = false;

        if (E_valid && !R_E.empty() && !t_E.empty()) {
            median_parallax = computeMedianParallax(R_E, t_E, pts1, pts2, K);
            if (debug) std::cerr << "Median parallax: " << median_parallax << "°" << std::endl;

            // Pure rotation criteria:
            // 1. Homography explains motion well (ratioH > 0.55)
            // 2. Very low parallax (< 1.0°)
            // 3. H is very rotation-like (ortho error < 0.02) - stricter threshold
            // 4. H-decomposed translation is near-zero
            double H_ortho_err = checkHomographyRotationLike(H, K);
            bool H_very_rotation_like = (H_ortho_err < 0.02);
            bool low_parallax = (median_parallax < 1.0);

            // Check H-decomposed translation magnitude
            double H_t_norm = 0.0;
            if (H_decomp_ok && !t_H.empty()) {
                H_t_norm = cv::norm(t_H);
            }

            if (debug) {
                std::cerr << "H orthogonality error: " << H_ortho_err
                          << " (very rotation-like: " << (H_very_rotation_like ? "yes" : "no") << ")" << std::endl;
                std::cerr << "H-decomposed t norm: " << H_t_norm << std::endl;
            }

            // Pure rotation if: very rotation-like H OR (low parallax AND poor E validation)
            if (ratioH > 0.55 && low_parallax && H_very_rotation_like) {
                pure_rotation_detected = true;
                if (debug) std::cerr << "PURE ROTATION DETECTED: ratioH=" << ratioH
                                     << " parallax=" << median_parallax << "° H_ortho=" << H_ortho_err
                                     << " goodRatio_E=" << goodRatio_E << std::endl;
            }
        }

        // If pure rotation, try to extract R from Homography and set t=0
        if (pure_rotation_detected && !H.empty()) {
            cv::Mat R_pure;
            if (extractRotationFromHomography(H, K, R_pure)) {
                R = R_pure.clone();
                t = cv::Mat::zeros(3, 1, CV_64F);
                if (debug) std::cerr << "Using Homography-derived rotation with zero translation" << std::endl;

                // Output and return early
                std::cout << std::fixed << std::setprecision(6);
                std::cout << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << std::endl;
                std::cout << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << std::endl;
                std::cout << R.at<double>(2,0) << " " << R.at<double>(2,1) << " " << R.at<double>(2,2) << std::endl;
                std::cout << "0.000000 0.000000 0.000000" << std::endl;
                return 0;
            } else if (debug) {
                std::cerr << "Failed to extract rotation from H, falling back to normal selection" << std::endl;
            }
        }

        // Selection logic using depth variance and direction agreement
        // Key insight: depth variance distinguishes 3D (trust E) from planar (trust H) scenes
        // Direction agreement determines confidence in the selection

        if (direction_agreement < 0.0 || std::abs(direction_agreement) < 0.5) {
            // E and H disagree significantly (opposite or very different directions)
            // Use depth variance to determine which is more reliable
            if (depth_cv > 0.5 && E_valid) {
                // High depth variance suggests 3D scene - trust E
                useH = false;
                if (debug) std::cerr << "Selection: E-H disagree, HIGH depth_cv=" << depth_cv << " -> E (3D scene)" << std::endl;
            } else if (H_valid) {
                // Low depth variance suggests planar scene - trust H (only if H is valid)
                useH = true;
                if (debug) std::cerr << "Selection: E-H disagree, LOW depth_cv=" << depth_cv << " -> H (planar scene)" << std::endl;
            } else if (E_valid) {
                useH = false;
                if (debug) std::cerr << "Selection: E-H disagree, H not available -> E" << std::endl;
            }
        }
        // E and H agree strongly (< 25° angle difference)
        else if (direction_agreement > 0.9) {
            // Strong agreement: E and H give almost same direction
            // Prefer E (more accurate when both work)
            if (E_valid) {
                useH = false;
                if (debug) std::cerr << "Selection: E-H strongly agree (cos=" << direction_agreement << ") -> E" << std::endl;
            } else if (H_valid) {
                useH = true;
                if (debug) std::cerr << "Selection: E-H agree but E invalid -> H" << std::endl;
            }
        }
        // Moderate agreement (0.5-0.9, or 25-60° angle)
        // Use depth variance as primary signal
        else if (E_valid && H_valid) {
            if (depth_cv > 0.5) {
                // High depth variance: 3D scene, prefer E
                useH = false;
                if (debug) std::cerr << "Selection: moderate agreement, HIGH depth_cv=" << depth_cv << " -> E (3D scene)" << std::endl;
            } else if (depth_cv < 0.4) {
                // Low depth variance: planar scene, prefer H
                useH = true;
                if (debug) std::cerr << "Selection: moderate agreement, LOW depth_cv=" << depth_cv << " -> H (planar scene)" << std::endl;
            } else {
                // Borderline depth variance: use ratioH as tiebreaker
                if (ratioH > 0.50) {
                    useH = true;
                    if (debug) std::cerr << "Selection: moderate, borderline depth_cv, ratioH=" << ratioH << " > 0.50 -> H" << std::endl;
                } else {
                    useH = false;
                    if (debug) std::cerr << "Selection: moderate, borderline depth_cv, ratioH=" << ratioH << " <= 0.50 -> E" << std::endl;
                }
            }
        }
        // Only E is valid
        else if (E_valid) {
            useH = false;
            if (debug) std::cerr << "Selection: only E valid -> E" << std::endl;
        }
        // Only H is valid
        else if (H_valid) {
            useH = true;
            if (debug) std::cerr << "Selection: only H valid -> H" << std::endl;
        }
        // Neither valid
        else {
            printZeroPose();
            return 0;
        }

        if (debug) {
            std::cerr << "Selected: " << (useH ? "HOMOGRAPHY" : "ESSENTIAL") << std::endl;
        }

        if (useH) {
            R = R_H.clone();
            t = t_H.clone();
            // No strict validation for H - trust the H/F score decision
        } else {
            R = R_E.clone();
            t = t_E.clone();
            // Final validation for E
            if (goodRatio_E < 0.5 || nGood_E < 8) {
                printZeroPose();
                return 0;
            }
        }
    }

    if (R.empty() || t.empty()) {
        printZeroPose();
        return 0;
    }

    // Output rotation matrix (3x3) and translation vector
    // t from recoverPose points from cam1 to cam2 origin
    // For trajectory (camera motion), we output -t
    std::cout << std::fixed << std::setprecision(6);

    // Rotation matrix (3 rows)
    std::cout << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << std::endl;
    std::cout << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << std::endl;
    std::cout << R.at<double>(2,0) << " " << R.at<double>(2,1) << " " << R.at<double>(2,2) << std::endl;

    // Translation vector
    std::cout << -t.at<double>(0) << " " << -t.at<double>(1) << " " << -t.at<double>(2) << std::endl;

    return 0;
}
