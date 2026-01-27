/*
 * Neural Visual Odometry - 2-Frame Relative Pose Estimation (SuperPoint + LightGlue)
 * Uses ONNX Runtime for inference, shared geometry backend with vo_submission.
 *
 * Usage: ./vo_neural <image1> <image2> [options]
 *   -f <focal>         Focal length (assumes fx=fy, cx=w/2, cy=h/2)
 *   -k <fx,fy,cx,cy>   Full camera intrinsics
 *   -d                 Debug output
 *   -M <models_dir>    Path to ONNX models directory (default: models/)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <numeric>

#include <onnxruntime_cxx_api.h>
#ifdef USE_CUDA
#include <onnxruntime_c_api.h>
#endif

#include "vo_geometry.hpp"

static bool tryAppendCUDA(Ort::SessionOptions& opts, bool debug) {
#ifdef USE_CUDA
    try {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        cuda_opts.arena_extend_strategy = 0;
        cuda_opts.gpu_mem_limit = SIZE_MAX;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        cuda_opts.do_copy_in_default_stream = 1;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        if (debug) std::cerr << "CUDA enabled" << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        if (debug) std::cerr << "CUDA not available, using CPU" << std::endl;
        return false;
    }
#else
    (void)opts; (void)debug;
    return false;
#endif
}

class SuperPointONNX {
public:
    SuperPointONNX(const std::string& model_path, bool debug = false)
        : debug_(debug) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        bool cuda_requested = tryAppendCUDA(opts, debug);
        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
        } catch (const Ort::Exception& e) {
            if (cuda_requested) {
                if (debug) std::cerr << "CUDA session failed, falling back to CPU" << std::endl;
                Ort::SessionOptions cpu_opts;
                cpu_opts.SetIntraOpNumThreads(4);
                cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), cpu_opts);
            } else {
                throw;
            }
        }
    }

    struct Result {
        std::vector<cv::Point2f> keypoints;
        std::vector<float> scores;
        std::vector<float> descriptors;
        int num_keypoints = 0;
        int input_h = 0, input_w = 0;
        float scale = 1.0f;
        int pad_h = 0, pad_w = 0;
    };

    Result detect(const cv::Mat& gray_img) {
        Result res;

        int orig_h = gray_img.rows, orig_w = gray_img.cols;
        float max_edge = static_cast<float>(std::max(orig_h, orig_w));
        res.scale = 1024.0f / max_edge;

        int new_h = static_cast<int>(std::round(orig_h * res.scale));
        int new_w = static_cast<int>(std::round(orig_w * res.scale));

        cv::Mat resized;
        cv::resize(gray_img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        int pad_h = (8 - (new_h % 8)) % 8;
        int pad_w = (8 - (new_w % 8)) % 8;
        res.pad_h = 0;
        res.pad_w = 0;

        cv::Mat padded;
        if (pad_h > 0 || pad_w > 0) {
            cv::copyMakeBorder(resized, padded, 0, pad_h, 0, pad_w,
                               cv::BORDER_CONSTANT, cv::Scalar(0));
        } else {
            padded = resized;
        }

        int H = padded.rows, W = padded.cols;
        res.input_h = H;
        res.input_w = W;

        std::vector<float> input_data(H * W);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                input_data[r * W + c] = padded.at<uchar>(r, c) / 255.0f;

        std::array<int64_t, 4> input_shape = {1, 1, H, W};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        const char* input_names[] = {"image"};
        const char* output_names[] = {"keypoints", "scores", "descriptors"};

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                      input_names, &input_tensor, 1,
                                      output_names, 3);

        auto kp_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto kp_shape = kp_info.GetShape();
        int N = static_cast<int>(kp_shape[1]);
        res.num_keypoints = N;

        const float* sc_data = outputs[1].GetTensorData<float>();
        const float* desc_data = outputs[2].GetTensorData<float>();

        auto desc_info = outputs[2].GetTensorTypeAndShapeInfo();
        auto desc_shape = desc_info.GetShape();
        int D = static_cast<int>(desc_shape[2]);

        res.keypoints.resize(N);
        res.scores.resize(N);
        res.descriptors.resize(N * D);

        auto kp_elem_type = kp_info.GetElementType();

        for (int i = 0; i < N; i++) {
            float x_sp, y_sp;
            if (kp_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                const int64_t* kp_data_i64 = outputs[0].GetTensorData<int64_t>();
                x_sp = static_cast<float>(kp_data_i64[i * 2 + 0]);
                y_sp = static_cast<float>(kp_data_i64[i * 2 + 1]);
            } else {
                const float* kp_data_f32 = outputs[0].GetTensorData<float>();
                x_sp = kp_data_f32[i * 2 + 0];
                y_sp = kp_data_f32[i * 2 + 1];
            }
            res.keypoints[i] = cv::Point2f((x_sp - res.pad_w) / res.scale,
                                            (y_sp - res.pad_h) / res.scale);
            res.scores[i] = sc_data[i];
        }

        std::memcpy(res.descriptors.data(), desc_data, N * D * sizeof(float));

        if (debug_)
            std::cerr << "SuperPoint: " << N << " kp, dim=" << D << std::endl;

        return res;
    }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "SuperPoint"};
    std::unique_ptr<Ort::Session> session_;
    bool debug_;
};

class LightGlueONNX {
public:
    LightGlueONNX(const std::string& model_path, bool debug = false)
        : debug_(debug), model_path_(model_path) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        has_cuda_ = tryAppendCUDA(opts, debug);
        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
        } catch (const Ort::Exception& e) {
            if (has_cuda_) {
                if (debug) std::cerr << "CUDA session failed, falling back to CPU" << std::endl;
                has_cuda_ = false;
                session_ = makeCpuSession();
            } else {
                throw;
            }
        }
    }

    struct Result {
        std::vector<std::pair<int, int>> matches;  // (idx0, idx1) pairs
        std::vector<float> match_scores;
    };

    Result match(const SuperPointONNX::Result& sp0,
                 const SuperPointONNX::Result& sp1) {
        Result res;

        int N = sp0.num_keypoints;
        int M = sp1.num_keypoints;

        if (N == 0 || M == 0) return res;

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        float W0 = static_cast<float>(sp0.input_w);
        float H0 = static_cast<float>(sp0.input_h);
        float W1 = static_cast<float>(sp1.input_w);
        float H1 = static_cast<float>(sp1.input_h);

        std::vector<float> kpts0_norm(N * 2);
        for (int i = 0; i < N; i++) {
            float x_sp = sp0.keypoints[i].x * sp0.scale + sp0.pad_w;
            float y_sp = sp0.keypoints[i].y * sp0.scale + sp0.pad_h;
            kpts0_norm[i * 2 + 0] = 2.0f * x_sp / W0 - 1.0f;
            kpts0_norm[i * 2 + 1] = 2.0f * y_sp / H0 - 1.0f;
        }

        std::vector<float> kpts1_norm(M * 2);
        for (int i = 0; i < M; i++) {
            float x_sp = sp1.keypoints[i].x * sp1.scale + sp1.pad_w;
            float y_sp = sp1.keypoints[i].y * sp1.scale + sp1.pad_h;
            kpts1_norm[i * 2 + 0] = 2.0f * x_sp / W1 - 1.0f;
            kpts1_norm[i * 2 + 1] = 2.0f * y_sp / H1 - 1.0f;
        }

        int D = static_cast<int>(sp0.descriptors.size()) / N;

        std::vector<float> desc0_data(sp0.descriptors.begin(), sp0.descriptors.end());
        std::vector<float> desc1_data(sp1.descriptors.begin(), sp1.descriptors.end());

        auto runInference = [&](Ort::Session& sess) {
            std::array<int64_t, 3> kp0_shape = {1, N, 2};
            std::array<int64_t, 3> kp1_shape = {1, M, 2};
            std::array<int64_t, 3> desc0_shape = {1, N, D};
            std::array<int64_t, 3> desc1_shape = {1, M, D};

            Ort::Value kp0_tensor = Ort::Value::CreateTensor<float>(
                memory_info, kpts0_norm.data(), kpts0_norm.size(),
                kp0_shape.data(), kp0_shape.size());
            Ort::Value kp1_tensor = Ort::Value::CreateTensor<float>(
                memory_info, kpts1_norm.data(), kpts1_norm.size(),
                kp1_shape.data(), kp1_shape.size());
            Ort::Value desc0_tensor = Ort::Value::CreateTensor<float>(
                memory_info, desc0_data.data(), desc0_data.size(),
                desc0_shape.data(), desc0_shape.size());
            Ort::Value desc1_tensor = Ort::Value::CreateTensor<float>(
                memory_info, desc1_data.data(), desc1_data.size(),
                desc1_shape.data(), desc1_shape.size());

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(kp0_tensor));
            input_tensors.push_back(std::move(kp1_tensor));
            input_tensors.push_back(std::move(desc0_tensor));
            input_tensors.push_back(std::move(desc1_tensor));

            const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
            const char* output_names[] = {"matches0", "mscores0"};

            return sess.Run(Ort::RunOptions{nullptr},
                            input_names, input_tensors.data(), 4,
                            output_names, 2);
        };

        std::vector<Ort::Value> outputs;
        try {
            outputs = runInference(*session_);
        } catch (const Ort::Exception& e) {
            if (has_cuda_) {
                if (debug_) std::cerr << "GPU OOM, retrying on CPU" << std::endl;
                if (!cpu_session_) cpu_session_ = makeCpuSession();
                outputs = runInference(*cpu_session_);
            } else {
                throw;
            }
        }

        auto m0_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto m0_shape = m0_info.GetShape();
        int match_len = static_cast<int>(m0_shape[1]);

        const int64_t* matches_data = outputs[0].GetTensorData<int64_t>();
        const float* mscores_data = outputs[1].GetTensorData<float>();

        for (int i = 0; i < match_len; i++) {
            int64_t j = matches_data[i];
            if (j >= 0) {
                res.matches.push_back({i, static_cast<int>(j)});
                res.match_scores.push_back(mscores_data[i]);
            }
        }

        if (debug_) {
            std::cerr << "LightGlue: " << res.matches.size() << " matches from "
                      << N << " + " << M << " keypoints" << std::endl;
        }

        return res;
    }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "LightGlue"};
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Session> cpu_session_;
    bool debug_;
    bool has_cuda_ = false;
    std::string model_path_;

    std::unique_ptr<Ort::Session> makeCpuSession() {
        Ort::SessionOptions cpu_opts;
        cpu_opts.SetIntraOpNumThreads(4);
        cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        return std::make_unique<Ort::Session>(env_, model_path_.c_str(), cpu_opts);
    }
};

int main(int argc, char** argv) {
    std::string img1_path, img2_path;
    std::string models_dir = "models";
    double user_focal = -1;
    double user_fx = -1, user_fy = -1, user_cx = -1, user_cy = -1;
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
        } else if (arg == "-M" && i + 1 < argc) {
            models_dir = argv[++i];
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
        std::cerr << "  -M <models_dir>    Path to ONNX models directory (default: models/)" << std::endl;
        std::cerr << "  -d                 Debug output" << std::endl;
        return 1;
    }

    bool has_full_calib = (user_fx > 0 && user_fy > 0 && user_cx > 0 && user_cy > 0);
    bool has_focal = (user_focal > 0);
    bool has_calib = has_full_calib || has_focal;

    cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images" << std::endl;
        return 1;
    }

    std::string sp_path = models_dir + "/superpoint.onnx";
    std::string lg_path = models_dir + "/superpoint_lightglue.onnx";

    SuperPointONNX superpoint(sp_path, debug);
    LightGlueONNX lightglue(lg_path, debug);

    auto sp0 = superpoint.detect(img1);
    auto sp1 = superpoint.detect(img2);

    if (sp0.num_keypoints < 30 || sp1.num_keypoints < 30) {
        if (debug) std::cerr << "Too few keypoints: " << sp0.num_keypoints
                             << " + " << sp1.num_keypoints << std::endl;
        vo_geometry::printZeroPose();
        return 0;
    }

    auto limitKeypoints = [](SuperPointONNX::Result& sp, int max_kp) {
        if (sp.num_keypoints <= max_kp) return;
        std::vector<int> indices(sp.num_keypoints);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return sp.scores[a] > sp.scores[b]; });
        indices.resize(max_kp);
        std::sort(indices.begin(), indices.end());

        int D = static_cast<int>(sp.descriptors.size()) / sp.num_keypoints;
        std::vector<cv::Point2f> new_kp(max_kp);
        std::vector<float> new_sc(max_kp);
        std::vector<float> new_desc(max_kp * D);
        for (int i = 0; i < max_kp; i++) {
            new_kp[i] = sp.keypoints[indices[i]];
            new_sc[i] = sp.scores[indices[i]];
            std::memcpy(&new_desc[i * D], &sp.descriptors[indices[i] * D], D * sizeof(float));
        }
        sp.keypoints = std::move(new_kp);
        sp.scores = std::move(new_sc);
        sp.descriptors = std::move(new_desc);
        sp.num_keypoints = max_kp;
    };

    const int MAX_KEYPOINTS = 2048;
    if (sp0.num_keypoints > MAX_KEYPOINTS || sp1.num_keypoints > MAX_KEYPOINTS) {
        if (debug) std::cerr << "Limiting keypoints: " << sp0.num_keypoints
                             << " + " << sp1.num_keypoints << " -> " << MAX_KEYPOINTS << std::endl;
        limitKeypoints(sp0, MAX_KEYPOINTS);
        limitKeypoints(sp1, MAX_KEYPOINTS);
    }

    auto lg_result = lightglue.match(sp0, sp1);

    if (lg_result.matches.size() < 10) {
        if (debug) std::cerr << "Too few matches: " << lg_result.matches.size() << std::endl;
        vo_geometry::printZeroPose();
        return 0;
    }

    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(lg_result.matches.size());
    pts2.reserve(lg_result.matches.size());

    for (const auto& [i0, i1] : lg_result.matches) {
        pts1.push_back(sp0.keypoints[i0]);
        pts2.push_back(sp1.keypoints[i1]);
    }

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
