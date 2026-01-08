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
- Eigen3
- PoseLib (included as submodule)
- C++17 compiler

## Build Instructions

### Linux (Ubuntu/Debian)
```bash
# Install dependencies
sudo apt update
sudo apt install libopencv-dev libeigen3-dev g++ cmake

# Clone with PoseLib
git clone https://github.com/salah-dev-stu/visual-odometry.git
cd visual-odometry
git clone https://github.com/PoseLib/PoseLib.git poselib_src

# Build PoseLib
cd poselib_src && mkdir -p build && cd build
cmake .. && make -j4
cd ../..

# Build VO
g++ -O3 -std=c++17 -o vo_submission src/vo_submission.cpp \
    -I poselib_src -I poselib_src/build/generated_headers \
    poselib_src/build/PoseLib/libPoseLib.a \
    $(pkg-config --cflags --libs opencv4 eigen3)
```

### macOS
```bash
# Install dependencies
brew install opencv eigen cmake

# Clone with PoseLib
git clone https://github.com/salah-dev-stu/visual-odometry.git
cd visual-odometry
git clone https://github.com/PoseLib/PoseLib.git poselib_src

# Build PoseLib
cd poselib_src && mkdir -p build && cd build
cmake .. && make -j4
cd ../..

# Build VO
g++ -O3 -std=c++17 -o vo_submission src/vo_submission.cpp \
    -I poselib_src -I poselib_src/build/generated_headers \
    poselib_src/build/PoseLib/libPoseLib.a \
    $(pkg-config --cflags --libs opencv4 eigen3)
```

### Windows

#### Build with vcpkg
```powershell
# 1. Install vcpkg (one-time setup)
cd %USERPROFILE%
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 2. Install dependencies
.\vcpkg install opencv4:x64-windows eigen3:x64-windows

# 3. Clone project with PoseLib
cd %USERPROFILE%
git clone https://github.com/salah-dev-stu/visual-odometry.git
cd visual-odometry
git clone https://github.com/PoseLib/PoseLib.git poselib_src

# 4. Build PoseLib
cd poselib_src
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=%USERPROFILE%\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build . --config Release
cd ..\..

# 5. Build VO (from Developer Command Prompt)
cl /EHsc /O2 /std:c++17 src\vo_submission.cpp ^
    /I poselib_src /I poselib_src\build\generated_headers ^
    /I %USERPROFILE%\vcpkg\installed\x64-windows\include ^
    /I %USERPROFILE%\vcpkg\installed\x64-windows\include\opencv4 ^
    /I %USERPROFILE%\vcpkg\installed\x64-windows\include\eigen3 ^
    /link poselib_src\build\PoseLib\Release\PoseLib.lib ^
    /LIBPATH:%USERPROFILE%\vcpkg\installed\x64-windows\lib ^
    opencv_core4.lib opencv_imgcodecs4.lib opencv_imgproc4.lib ^
    opencv_features2d4.lib opencv_calib3d4.lib

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
| `-s <scale>` | Image scale factor (0.25, 0.5, 1.0) | Auto (1.0 for auto-focal, 0.5 for known focal) |
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
| Auto focal (no flags) | ~0.39s | Default, no calibration needed |
| Known focal, scale=0.5 | ~0.25s | When calibration known |
| Known focal, scale=1.0 | ~0.50s | High precision |
| Known focal, scale=0.25 | ~0.10s | Real-time embedded |

## Acknowledgments

This project was developed with assistance from [Claude Code](https://claude.ai/code) AI.
