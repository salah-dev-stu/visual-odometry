# Algorithm Documentation

Technical details of the 2-frame visual odometry implementation.

## Pipeline Overview

```
Input Images → Downscale → ORB Detection → Feature Matching → Pose Estimation → Output R,t
                              ↓                   ↓
                         2000/1000/500        Ratio Test +
                          features            Cross-Check
                                                  ↓
                                    ┌─────────────┴─────────────┐
                                    ↓                           ↓
                              Known Focal                  Auto-Focal
                                    ↓                           ↓
                              H/E Selection            PoseLib CVPR'24
                                    ↓                     Estimation
                              Best Pose                       ↓
                                    └───────────┬─────────────┘
                                                ↓
                                           Validation
                                                ↓
                                         Output R, t
```

## 1. Feature Detection and Matching

### ORB Features
- **Detector**: OpenCV ORB (Oriented FAST and Rotated BRIEF)
- **Feature count**: Scales with image resolution
  - Full resolution (scale=1.0): 2000 features
  - Half resolution (scale=0.5): 1000 features
  - Quarter resolution (scale=0.25): 500 features

### Matching Strategy
- **Matcher**: Brute-force with Hamming distance (for binary ORB descriptors)
- **Lowe's Ratio Test**: Keep matches where `best_distance < 0.75 * second_best_distance`
- **Cross-Check**: Match from image1→image2 and image2→image1, keep only mutual matches

This combination rejects ~90% of false matches, leaving high-quality correspondences.

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

When the camera rotates without translation, the Essential matrix becomes degenerate. We detect this case using:

1. **Homography Orthogonality Check**: For pure rotation, H = K·R·K⁻¹, so R = K⁻¹·H·K should be orthogonal
   - Compute `R_approx = K⁻¹ * H * K`
   - Normalize by determinant: `R_approx = R_approx / det(R_approx)^(1/3)`
   - Check orthogonality: `error = ||R^T * R - I||`
   - If error < 0.02: H is "very rotation-like"

2. **Low Parallax**: Median parallax angle from triangulated points < 1.0°

3. **H Dominance**: ratioH > 0.55 (Homography explains motion well)

When all three conditions are met:
- Extract rotation directly from H: `R = K⁻¹ * H * K`, then force orthogonality via SVD
- Output zero translation (pure rotation has no translation component)

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

5. **Pure Rotation**: When the camera only rotates (no translation), the Essential matrix becomes degenerate. The code detects this using Homography orthogonality checks and outputs rotation-only pose with zero translation (see Section 3.4).

## References

1. Kocur, Kyselica, Kukelova. "Robust Self-calibration of Focal Lengths from the Fundamental Matrix." CVPR 2024.

2. Barath, Matas. "Graph-Cut RANSAC." CVPR 2018. (Foundation for USAC)

3. Barath et al. "MAGSAC: Marginalizing Sample Consensus." CVPR 2019.

4. Hartley, Zisserman. "Multiple View Geometry in Computer Vision." Cambridge University Press.
