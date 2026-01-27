# Algorithm Documentation

Technical details of the 2-frame visual odometry implementation. The system has two frontends (ORB or SuperPoint+LightGlue) sharing a common geometry backend.

## Pipeline Overview

```
                    ┌─── ORB Frontend (vo_submission) ───┐
                    │  Downscale → ORB → BF + Ratio Test │
                    │              + Cross-Check          │
Input Images ───────┤                                     ├──→ Geometry Backend → Output R,t
                    │                                     │
                    ├─── Neural Frontend (vo_neural) ─────┤
                    │  SuperPoint (ONNX) → LightGlue      │
                    │  (ONNX) → Matched Points            │
                    │  Depth Anything V2 (ONNX) → Depth   │
                    │  Back-project → PnP → R, t          │
                    │  (fallback to E/H if PnP fails)     │
                    └─────────────────────────────────────┘

Geometry Backend (E/H path, used by ORB and as neural fallback):
    Matched Points ──→ Known Focal? ──→ H/E Selection ──→ Validation ──→ Output R,t
                              │
                              └──→ Auto-Focal (PoseLib CVPR'24) ──→ E path only
```

## 1. Feature Detection and Matching

### 1a. ORB Frontend (`vo_submission`)

- **Detector**: OpenCV ORB (Oriented FAST and Rotated BRIEF)
- **Feature count**: Scales with image resolution (2000 / 1000 / 500)
- **Matcher**: Brute-force with Hamming distance
- **Lowe's Ratio Test**: Keep matches where `best_distance < 0.75 * second_best_distance`
- **Cross-Check**: Match bidirectionally, keep only mutual matches
- Rejects ~90% of false matches

### 1b. Neural Frontend (`vo_neural`)

- **Detector**: SuperPoint (ONNX, 256-dim descriptors)
  - Input resized to 1024px longest edge, padded to multiple of 8
  - Keypoints capped at 2048 (top by score) to limit GPU memory
  - Outputs mapped back to original image coordinates
- **Matcher**: LightGlue (ONNX, attention-based)
  - Keypoints normalized to [-1, 1] before matching
  - GPU inference with automatic CPU fallback on OOM
- Produces fewer but more accurate correspondences than ORB

### 1c. Monocular Depth Estimation (`vo_neural`)

- **Model**: Depth Anything V2 ViT-S (ONNX, ~101 MB)
- **Input**: BGR image resized to 518x518, RGB-converted, normalized with ImageNet mean/std
  - mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
- **Output**: Relative inverse depth map (higher values = closer), resized back to original resolution
- **Usage**: Only run on frame 1 (the frame providing 3D points for PnP)
- GPU inference with automatic CPU fallback

The depth map is used to back-project matched keypoints from frame 1 into 3D space, enabling PnP-based pose estimation (Section 3.5).

## 2. Auto-Focal Estimation (No Calibration Mode)

When no focal length is provided, we use the **CVPR 2024 iterative method** from PoseLib:

> Kocur, Kyselica, Kukelova, "Robust Self-calibration of Focal Lengths from the Fundamental Matrix"

### Process:
1. Estimate Fundamental matrix F using USAC_MAGSAC
2. Initialize focal length guess: `f = 0.85 * image_width`
3. Iteratively refine focal length (up to 50 iterations)
4. Use average of estimated f1, f2 (should be equal for same camera)

### Why This Works:
The fundamental matrix encodes the epipolar geometry between two views. For a calibrated camera, `E = K^T * F * K`. The iterative method finds the focal length that best satisfies the Essential matrix constraints.

## 3. Pose Estimation (Known Calibration Mode)

With known calibration, we compute both **Essential** and **Homography** matrices and select the better one.

### 3.1 Essential Matrix Path
- **Estimation**: `cv::findEssentialMat` with USAC_MAGSAC
- **Decomposition**: `cv::recoverPose` (returns R, t with cheirality check)
- **Fallback**: If direct E fails, compute `E = K^T * F * K`

### 3.2 Homography Matrix Path
- **Estimation**: `cv::findHomography` with USAC_MAGSAC
- **Decomposition**: `cv::decomposeHomographyMat` (returns up to 4 solutions)
- **Selection**: Choose solution with most points having positive depth in both cameras

### 3.3 H/E Selection Logic

The key challenge is selecting between Essential (better for 3D scenes) and Homography (better for planar scenes).

#### Scoring:
- **Homography Score**: Sum of symmetric transfer errors below threshold
- **Fundamental Score**: Sum of Sampson errors below threshold
- **Ratio**: `ratioH = scoreH / (scoreH + scoreF)`

#### Selection Criteria:
1. **Depth Variance**: Computed from triangulated points
   - High variance (CV > 0.5) → 3D scene → prefer Essential
   - Low variance (CV < 0.4) → planar scene → prefer Homography

2. **Direction Agreement**: Cosine angle between t_E and t_H
   - Strong agreement (cos > 0.9) → prefer Essential (more accurate)
   - Disagreement (cos < 0.5) → use depth variance to decide

3. **Validation Thresholds**:
   - Essential: ≥70% of points pass triangulation checks
   - Homography: ≥25% of points pass triangulation checks (must pass validation to be selected)

### 3.4 Pure Rotation Detection

When the camera rotates without translation, the Essential matrix becomes degenerate. Two detection paths:

**Path A (E valid but degenerate):**
1. **H Orthogonality Check**: `R_approx = K⁻¹ * H * K`, normalize by `det^(1/3)`, check `||R^T R - I|| < 0.02`
2. **Low Parallax**: Median parallax from triangulated points < 1.0°
3. **H Dominance**: ratioH > 0.55

**Path B (E completely failed):**
When `recoverPose` returns 0 inliers despite many matches (common with precise neural matches on small-baseline pairs), check if H is rotation-like with a relaxed threshold (`||R^T R - I|| < 0.05`).

When detected:
- Extract rotation from H via SVD orthogonalization
- Output zero translation

### 3.5 Depth-Based PnP Path (Neural Frontend Only)

When Depth Anything V2 is available and camera calibration is known, `vo_neural` uses a PnP-based pose estimation path before falling back to E/H:

#### Process:
1. **Depth estimation**: Run Depth Anything V2 on frame 1 to get a relative inverse depth map
2. **Back-projection**: For each matched keypoint `(u, v)` in frame 1:
   - Sample depth map with bilinear interpolation: `inv_d = depth_map(u, v)`
   - Convert inverse depth to depth: `d = 1 / inv_d`
   - Back-project to 3D: `X = (u - cx) * d / fx`, `Y = (v - cy) * d / fy`, `Z = d`
   - Skip points with invalid depth (`< 1e-3` or `> 1e4`)
3. **PnP solve**: `cv::solvePnPRansac(pts3D, pts2_2D, K, distCoeffs=0)` with SOLVEPNP_ITERATIVE, 300 iterations, 4px reprojection threshold
4. **Validation**: Reject if inlier ratio < 30%, inliers < 8, or mean reprojection error > 5px
5. **Normalization**: Translation is normalized to unit length (depth is relative, not metric)
6. **Fallback**: If PnP fails, fall back to E/H geometry path

#### Why PnP helps:
- **Degenerate geometry**: Pure rotation and planar scenes cause Essential matrix to fail. PnP uses 3D structure from depth, avoiding these degeneracies.
- **Robustness**: The depth network provides scene structure even when two-view geometry is ambiguous (small baselines, co-planar points).
- **Translation direction**: Even though depth is relative (not metric), the *direction* of translation recovered by PnP is correct, which is what matters for trajectory estimation.

#### Limitations:
- Depth is relative, not metric. Scale is consistent within a frame but arbitrary across frames.
- Depth prediction can be unreliable on reflective surfaces, sky regions, or textureless areas.
- Adds inference cost (~13ms GPU, ~100ms CPU) for the depth model.

## 4. Pose Validation

Each pose candidate is validated by triangulating points and checking:

### Triangulation (DLT Method)
For each point correspondence:
1. Build 4x4 matrix A from projection equations
2. Solve via SVD: `X = last row of V^T`
3. Dehomogenize: `X3D = X / X[3]`

### Validation Checks:
1. **Cheirality**: Depth must be positive in both cameras
2. **Reprojection Error**: Must be < 4 pixels in both views
3. **Depth Statistics**: Compute median and variance for scene type detection

## 5. USAC_MAGSAC

All robust estimation uses OpenCV's USAC_MAGSAC:

- **USAC**: Universal RANSAC framework with multiple improvements
- **MAGSAC**: Marginalizing Sample Consensus
  - σ-consensus: weights samples by quality instead of binary in/out
  - More robust to outlier contamination than standard RANSAC
  - Better threshold adaptation

## 6. Output Format

```
R11 R12 R13    ← Rotation matrix row 1
R21 R22 R23    ← Rotation matrix row 2
R31 R32 R33    ← Rotation matrix row 3
tx ty tz       ← Translation vector (unit length)
```

**Note**: Translation is negated from `recoverPose` output. The function returns t pointing from camera 1 to camera 2 in camera 1's frame. We negate it for trajectory computation (camera motion direction).

## 7. Known Limitations

1. **Scale Ambiguity**: Monocular VO cannot recover absolute scale. Translation is unit length.

2. **No Loop Closure**: 2-frame constraint means no correction when revisiting locations.

3. **No Bundle Adjustment**: Each frame pair is independent; no global optimization.

4. **Drift**: Errors accumulate over long sequences due to the above limitations.

5. **Pure Rotation**: When the camera only rotates (no translation), the Essential matrix becomes degenerate. Detected via Homography orthogonality checks and handled by extracting rotation from H (see Section 3.4).

## References

1. Kocur, Kyselica, Kukelova. "Robust Self-calibration of Focal Lengths from the Fundamental Matrix." CVPR 2024.

2. Barath, Matas. "Graph-Cut RANSAC." CVPR 2018. (Foundation for USAC)

3. Barath et al. "MAGSAC: Marginalizing Sample Consensus." CVPR 2019.

4. Hartley, Zisserman. "Multiple View Geometry in Computer Vision." Cambridge University Press.

5. DeTone, Malisiewicz, Rabinovich. "SuperPoint: Self-Supervised Interest Point Detection and Description." CVPR 2018 Workshop.

6. Lindenberger, Sarlin, Pollefeys. "LightGlue: Local Feature Matching at Light Speed." ICCV 2023.

7. Yang et al. "Depth Anything V2." NeurIPS 2024.
