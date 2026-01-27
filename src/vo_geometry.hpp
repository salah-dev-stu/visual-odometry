#pragma once
/*
 * vo_geometry.hpp — Shared geometry pipeline for Visual Odometry
 *
 * Contains:
 *   - Utility scoring / validation functions
 *   - Full E/H selection + pure-rotation detection pipeline
 *   - Pose output helpers
 *
 * Used by both vo_submission (ORB frontend) and vo_neural (ONNX frontend).
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iomanip>

// PoseLib includes for CVPR 2024 focal estimation
#include <PoseLib/misc/decompositions.h>
#include <PoseLib/misc/colmap_models.h>
#include <Eigen/Dense>

namespace vo_geometry {

struct GeometryConfig {
    double fx = -1, fy = -1, cx = -1, cy = -1;
    bool has_full_calib = false;
    bool has_focal = false;
    bool has_calib = false;
    bool debug = false;
    int img_width = 0;
    int img_height = 0;
};

struct PoseResult {
    cv::Mat R, t;
    bool valid = false;
};

inline double computeSampsonError(const cv::Point2f& pt1, const cv::Point2f& pt2,
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

inline double computeHomographyScore(const std::vector<cv::Point2f>& pts1,
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

inline double computeFundamentalScore(const std::vector<cv::Point2f>& pts1,
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

inline double computeMedianParallax(const cv::Mat& R, const cv::Mat& t,
                                     const std::vector<cv::Point2f>& pts1,
                                     const std::vector<cv::Point2f>& pts2,
                                     const cv::Mat& K) {
    cv::Mat K_inv = K.inv();
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P2(3, 4, CV_64F);
    R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    t.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    cv::Mat O1 = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat O2 = -R.t() * t;

    std::vector<double> parallaxes;

    for (size_t i = 0; i < pts1.size(); i++) {
        cv::Mat p1_h = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2_h = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat p1_n = K_inv * p1_h;
        cv::Mat p2_n = K_inv * p2_h;

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

        if (X3D.at<double>(2) <= 0) continue;
        cv::Mat X3D_c2 = R * X3D + t;
        if (X3D_c2.at<double>(2) <= 0) continue;

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

inline double checkHomographyRotationLike(const cv::Mat& H, const cv::Mat& K) {
    if (H.empty()) return 1e10;

    cv::Mat K_inv = K.inv();
    cv::Mat R_approx = K_inv * H * K;

    double det = cv::determinant(R_approx);
    if (std::abs(det) < 1e-10) return 1e10;
    if (det < 0) det = -det;
    double scale = std::pow(det, 1.0/3.0);
    R_approx = R_approx / scale;

    cv::Mat RtR = R_approx.t() * R_approx;
    return cv::norm(RtR - cv::Mat::eye(3, 3, CV_64F));
}

inline bool extractRotationFromHomography(const cv::Mat& H, const cv::Mat& K, cv::Mat& R_out) {
    if (H.empty()) return false;

    cv::Mat K_inv = K.inv();
    cv::Mat R_approx = K_inv * H * K;

    double det = cv::determinant(R_approx);
    if (det < 0) {
        R_approx = -R_approx;
        det = -det;
    }

    double scale = std::pow(det, 1.0/3.0);
    R_approx = R_approx / scale;

    cv::Mat w, u, vt;
    cv::SVD::compute(R_approx, w, u, vt);
    R_out = u * vt;

    if (cv::determinant(R_out) < 0) {
        R_out = -R_out;
    }

    cv::Mat I = R_out * R_out.t();
    double ortho_err = cv::norm(I - cv::Mat::eye(3, 3, CV_64F));
    return ortho_err < 0.1;
}

inline bool selectBestHomographyPose(const std::vector<cv::Point2f>& pts1,
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

inline void printZeroPose() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "1.000000 0.000000 0.000000" << std::endl;
    std::cout << "0.000000 1.000000 0.000000" << std::endl;
    std::cout << "0.000000 0.000000 1.000000" << std::endl;
    std::cout << "0.000000 0.000000 0.000000" << std::endl;
}

inline int validatePose(const cv::Mat& R, const cv::Mat& t,
                         const std::vector<cv::Point2f>& pts1,
                         const std::vector<cv::Point2f>& pts2,
                         const cv::Mat& K, double reproj_thresh = 4.0,
                         double* depth_variance = nullptr, double* median_depth = nullptr) {
    cv::Mat K_inv = K.inv();
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P2(3, 4, CV_64F);
    R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    t.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    int nGood = 0;
    std::vector<double> depths;

    for (size_t i = 0; i < pts1.size(); i++) {
        cv::Mat p1_h = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2_h = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat p1_n = K_inv * p1_h;
        cv::Mat p2_n = K_inv * p2_h;

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

        double z1 = X3D.at<double>(2);
        if (z1 <= 0) continue;

        cv::Mat X3D_c2 = R * X3D + t;
        double z2 = X3D_c2.at<double>(2);
        if (z2 <= 0) continue;

        double invZ1 = 1.0 / z1;
        double u1 = K.at<double>(0, 0) * X3D.at<double>(0) * invZ1 + K.at<double>(0, 2);
        double v1 = K.at<double>(1, 1) * X3D.at<double>(1) * invZ1 + K.at<double>(1, 2);
        double err1 = (u1 - pts1[i].x) * (u1 - pts1[i].x) + (v1 - pts1[i].y) * (v1 - pts1[i].y);
        if (err1 > reproj_thresh) continue;

        double invZ2 = 1.0 / z2;
        double u2 = K.at<double>(0, 0) * X3D_c2.at<double>(0) * invZ2 + K.at<double>(0, 2);
        double v2 = K.at<double>(1, 1) * X3D_c2.at<double>(1) * invZ2 + K.at<double>(1, 2);
        double err2 = (u2 - pts2[i].x) * (u2 - pts2[i].x) + (v2 - pts2[i].y) * (v2 - pts2[i].y);
        if (err2 > reproj_thresh) continue;

        nGood++;
        depths.push_back(z1);
    }

    if (depth_variance && median_depth && !depths.empty()) {
        std::sort(depths.begin(), depths.end());
        *median_depth = depths[depths.size() / 2];

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
inline double estimateFocalPoseLib(const std::vector<cv::Point2f>& pts1,
                                    const std::vector<cv::Point2f>& pts2,
                                    double cx, double cy, double initial_focal,
                                    bool debug = false) {
    cv::Mat F_cv = cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, 1.0, 0.999);
    if (F_cv.empty() || F_cv.rows != 3) {
        if (debug) std::cerr << "F matrix estimation failed" << std::endl;
        return initial_focal;
    }

    Eigen::Matrix3d F;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            F(i, j) = F_cv.at<double>(i, j);
        }
    }

    poselib::Camera camera1_prior(poselib::SimplePinholeCameraModel::model_id,
                                   std::vector<double>{initial_focal, cx, cy}, -1, -1);
    poselib::Camera camera2_prior(poselib::SimplePinholeCameraModel::model_id,
                                   std::vector<double>{initial_focal, cx, cy}, -1, -1);

    auto [camera1_out, camera2_out, iters] =
        poselib::focals_from_fundamental_iterative(F, camera1_prior, camera2_prior, 50);

    double f1 = camera1_out.focal();
    double f2 = camera2_out.focal();

    if (debug) {
        std::cerr << "PoseLib focal: f1=" << f1 << " f2=" << f2
                  << " iters=" << iters << " init=" << initial_focal << std::endl;
    }

    double min_focal = initial_focal * 0.2;
    double max_focal = initial_focal * 3.0;
    if (f1 > min_focal && f2 > min_focal && f1 < max_focal && f2 < max_focal) {
        return (f1 + f2) / 2.0;
    }

    return initial_focal;
}

inline void outputPose(const cv::Mat& R, const cv::Mat& t) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << std::endl;
    std::cout << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << std::endl;
    std::cout << R.at<double>(2,0) << " " << R.at<double>(2,1) << " " << R.at<double>(2,2) << std::endl;
    std::cout << -t.at<double>(0) << " " << -t.at<double>(1) << " " << -t.at<double>(2) << std::endl;
}

inline PoseResult estimatePose(const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& pts2,
                                const GeometryConfig& config) {
    PoseResult result;
    result.valid = false;

    double fx = config.fx, fy = config.fy, cx = config.cx, cy = config.cy;
    bool debug = config.debug;

    cv::Mat R, t;

    if (config.has_calib) {
        // Use provided calibration
    } else {
        // Use CVPR 2024 iterative method from PoseLib for focal estimation
        double initial_focal = config.img_width * 0.85;
        fx = fy = estimateFocalPoseLib(pts1, pts2, cx, cy, initial_focal, debug);

        cv::Mat K_auto = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K_auto, cv::USAC_MAGSAC, 0.999, 1.0, mask);

        if (E.empty() || E.rows != 3) {
            return result;
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
            return result;
        }

        int nGood = validatePose(R, t, pts1_in, pts2_in, K_auto);
        double goodRatio = (double)nGood / pts1_in.size();
        if (debug) {
            std::cerr << "Auto-focal validation: nGood=" << nGood << "/" << pts1_in.size()
                      << " ratio=" << goodRatio << std::endl;
        }
        if (goodRatio < 0.7 || nGood < 8) {
            return result;
        }

        result.R = R.clone();
        result.t = t.clone();
        result.valid = true;
        return result;
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Calibrated path: E/H selection
    cv::Mat H_mask, F_mask;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::USAC_MAGSAC, 3.0, H_mask);
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, 1.0, 0.999, F_mask);

    if (F.empty()) {
        return result;
    }

    double scoreH = computeHomographyScore(pts1, pts2, H, 5.99 * 2);
    double scoreF = computeFundamentalScore(pts1, pts2, F, 3.84 * 2);
    double ratioH = scoreH / (scoreH + scoreF + 1e-10);

    if (debug) {
        std::cerr << "H/F: scoreH=" << scoreH << " scoreF=" << scoreF
                  << " ratioH=" << ratioH << std::endl;
    }

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
            E_valid = (goodRatio_E >= 0.3 && nGood_E >= 5);
        }
        if (debug) {
            std::cerr << "E path: inliers=" << inliers << " nGood=" << nGood_E
                      << " ratio=" << goodRatio_E << " valid=" << E_valid
                      << " depth_var=" << depth_variance_E << " median_depth=" << median_depth_E << std::endl;
        }
    }

    // Try H matrix path if H looks reasonable, or if E failed (fallback)
    bool H_decomp_ok = false;
    if (!H.empty() && (ratioH > 0.40 || !E_valid)) {
        std::vector<cv::Mat> Rs, ts, normals;
        int solutions = cv::decomposeHomographyMat(H, K, Rs, ts, normals);

        if (solutions > 0 && selectBestHomographyPose(pts1, pts2, K, Rs, ts, R_H, t_H)) {
            H_decomp_ok = true;
            nGood_H = validatePose(R_H, t_H, pts1, pts2, K);
            goodRatio_H = (double)nGood_H / pts1.size();
            H_valid = (goodRatio_H >= 0.25 && nGood_H >= 5);
        }
        if (debug) {
            std::cerr << "H path: decomp=" << H_decomp_ok << " nGood=" << nGood_H
                      << " ratio=" << goodRatio_H << " valid=" << H_valid << std::endl;
        }
    }

    // Smart H/E selection
    bool useH = false;

    double direction_agreement = 0.0;
    if (E_valid && H_decomp_ok && !t_E.empty() && !t_H.empty()) {
        cv::Mat t_E_norm = t_E / cv::norm(t_E);
        cv::Mat t_H_norm = t_H / cv::norm(t_H);
        direction_agreement = t_E_norm.dot(t_H_norm);
        if (debug) std::cerr << "E-H direction agreement: " << direction_agreement
                             << " (angle: " << std::acos(std::abs(direction_agreement)) * 180.0 / CV_PI << "°)" << std::endl;
    }

    double depth_cv = 0.0;
    if (median_depth_E > 0.01) {
        depth_cv = std::sqrt(depth_variance_E) / median_depth_E;
    }
    if (debug) std::cerr << "Depth CV (std/median): " << depth_cv << std::endl;

    double median_parallax = 0.0;
    bool pure_rotation_detected = false;

    if (E_valid && !R_E.empty() && !t_E.empty()) {
        median_parallax = computeMedianParallax(R_E, t_E, pts1, pts2, K);
        if (debug) std::cerr << "Median parallax: " << median_parallax << "°" << std::endl;

        double H_ortho_err = checkHomographyRotationLike(H, K);
        bool H_very_rotation_like = (H_ortho_err < 0.02);
        bool low_parallax = (median_parallax < 1.0);

        double H_t_norm = 0.0;
        if (H_decomp_ok && !t_H.empty()) {
            H_t_norm = cv::norm(t_H);
        }

        if (debug) {
            std::cerr << "H orthogonality error: " << H_ortho_err
                      << " (very rotation-like: " << (H_very_rotation_like ? "yes" : "no") << ")" << std::endl;
            std::cerr << "H-decomposed t norm: " << H_t_norm << std::endl;
        }

        if (ratioH > 0.55 && low_parallax && H_very_rotation_like) {
            pure_rotation_detected = true;
            if (debug) std::cerr << "Pure rotation: ratioH=" << ratioH
                                 << " parallax=" << median_parallax << "°" << std::endl;
        }
    }

    if (!E_valid && !H.empty() && pts1.size() >= 30) {
        double H_ortho_err = checkHomographyRotationLike(H, K);
        if (H_ortho_err < 0.05) {
            pure_rotation_detected = true;
            if (debug) std::cerr << "Pure rotation (E failed): H_ortho=" << H_ortho_err << std::endl;
        }
    }

    if (pure_rotation_detected && !H.empty()) {
        cv::Mat R_pure;
        if (extractRotationFromHomography(H, K, R_pure)) {
            result.R = R_pure.clone();
            result.t = cv::Mat::zeros(3, 1, CV_64F);
            result.valid = true;
            if (debug) std::cerr << "Output: pure rotation from H" << std::endl;
            return result;
        }
    }

    if (direction_agreement < 0.0 || std::abs(direction_agreement) < 0.5) {
        if (depth_cv > 0.5 && E_valid) useH = false;
        else if (H_valid) useH = true;
        else if (E_valid) useH = false;
    } else if (direction_agreement > 0.9) {
        if (E_valid) useH = false;
        else if (H_valid) useH = true;
    } else if (E_valid && H_valid) {
        if (depth_cv > 0.5) useH = false;
        else if (depth_cv < 0.4) useH = true;
        else useH = (ratioH > 0.50);
    } else if (E_valid) {
        useH = false;
    } else if (H_valid) {
        useH = true;
    } else {
        return result;
    }

    if (debug) std::cerr << "Selected: " << (useH ? "H" : "E") << std::endl;

    if (useH) {
        R = R_H.clone();
        t = t_H.clone();
    } else {
        R = R_E.clone();
        t = t_E.clone();
        if (goodRatio_E < 0.5 || nGood_E < 8) {
            return result;
        }
    }

    if (R.empty() || t.empty()) {
        return result;
    }

    result.R = R.clone();
    result.t = t.clone();
    result.valid = true;
    return result;
}

} // namespace vo_geometry
