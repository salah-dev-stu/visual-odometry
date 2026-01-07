# Visual Odometry - 2-Frame Pose Estimation

Estimates relative camera pose (6DOF) between two images using ORB features and USAC_MAGSAC robust estimation.

## Output Format
```
roll pitch yaw tx ty tz
```
- Rotation: Euler angles in degrees (ZYX convention)
- Translation: unit vector (scale is unknown in monocular VO)

## Requirements

- OpenCV 4.x
- C++17 compiler

## Build Instructions

### Linux (Ubuntu/Debian)
```bash
# Install OpenCV
sudo apt update
sudo apt install libopencv-dev g++

# Build
g++ -O3 -std=c++17 -o vo_submission src/vo_submission.cpp $(pkg-config --cflags --libs opencv4)
```

### macOS
```bash
# Install OpenCV via Homebrew
brew install opencv

# Build
g++ -O3 -std=c++17 -o vo_submission src/vo_submission.cpp $(pkg-config --cflags --libs opencv4)
```

### Windows

#### Prerequisites
1. Install **Visual Studio Build Tools** (with "Desktop development with C++" workload):
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Run installer and select "Desktop development with C++"

#### Build with vcpkg
```powershell
# 1. Install vcpkg (one-time setup, from your home directory)
cd %USERPROFILE%
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 2. Install OpenCV (takes 15-20 minutes)
.\vcpkg install opencv4:x64-windows

# 3. Clone this project
cd %USERPROFILE%
git clone https://github.com/salah-dev-stu/visual-odometry.git
cd visual-odometry

# 4. Open Developer Command Prompt (sets up compiler environment)
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# 5. Build
cl /EHsc /O2 /std:c++17 src\vo_submission.cpp /I %USERPROFILE%\vcpkg\installed\x64-windows\include\opencv4 /link /LIBPATH:%USERPROFILE%\vcpkg\installed\x64-windows\lib opencv_core4.lib opencv_imgcodecs4.lib opencv_imgproc4.lib opencv_features2d4.lib opencv_calib3d4.lib

# 6. Copy required DLLs
copy %USERPROFILE%\vcpkg\installed\x64-windows\bin\*.dll .
```

## Usage

```bash
./vo_submission <image1> <image2> [-f focal_length] [-s scale] [-m matches_file]
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-f <focal>` | Camera focal length in pixels (original image resolution) | Auto-estimate |
| `-s <scale>` | Image scale factor (0.25, 0.5, 1.0) | 0.5 |
| `-m <file>` | Output matched points to file (for visualization) | None |

### Examples

```bash
# Basic usage (auto focal, half resolution - fastest)
./vo_submission frame001.jpg frame002.jpg

# With known focal length (faster, more accurate)
./vo_submission frame001.jpg frame002.jpg -f 525.0

# Full resolution (slower, use for high-precision)
./vo_submission frame001.jpg frame002.jpg -f 525.0 -s 1.0

# Quarter resolution (fastest, for real-time on embedded)
./vo_submission frame001.jpg frame002.jpg -s 0.25
```

### Performance (Raspberry Pi 4)

| Mode | Time | Use Case |
|------|------|----------|
| Auto focal, scale=0.5 | ~0.50s | Default |
| Known focal, scale=0.5 | ~0.40s | When calibration known |
| Known focal, scale=1.0 | ~0.75s | High precision |
| Known focal, scale=0.25 | ~0.15s | Real-time embedded |

## Acknowledgments

This project was developed with assistance from [Claude Code](https://claude.ai/code) AI.
