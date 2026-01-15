# Visual Odometry - 2-Frame Pose Estimation

[![Build and Test](https://github.com/salah-dev-stu/visual-odometry/actions/workflows/build.yml/badge.svg)](https://github.com/salah-dev-stu/visual-odometry/actions/workflows/build.yml)

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

### Linux (Ubuntu/Debian) ✓ Tested

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev libeigen3-dev

# Clone repository
git clone https://github.com/salah-dev-stu/visual-odometry.git
cd visual-odometry

# Clone and build PoseLib
git clone https://github.com/PoseLib/PoseLib.git poselib_src
cd poselib_src && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..

# Build with CMake
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### macOS ✓ Tested

```bash
# Install dependencies
brew install opencv eigen

# Clone repository
git clone https://github.com/salah-dev-stu/visual-odometry.git
cd visual-odometry

# Clone and build PoseLib
git clone https://github.com/PoseLib/PoseLib.git poselib_src
cd poselib_src && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=$(brew --prefix eigen)/include/eigen3
make -j$(sysctl -n hw.ncpu)
cd ../..

# Build with CMake
mkdir -p build && cd build
cmake .. -DEIGEN3_INCLUDE_DIR=$(brew --prefix eigen)/include/eigen3
make -j$(sysctl -n hw.ncpu)
```

### Windows ✓ Tested

```powershell
# Install OpenCV via Chocolatey (run as Administrator)
choco install opencv -y

# Download and install Eigen
Invoke-WebRequest -Uri "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" -OutFile eigen.zip
Expand-Archive eigen.zip -DestinationPath .
mkdir eigen3_install
cd eigen-3.4.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$PWD/../../eigen3_install"
cmake --install .
cd ../..

# Clone repository
git clone https://github.com/salah-dev-stu/visual-odometry.git
cd visual-odometry

# Clone and build PoseLib
git clone https://github.com/PoseLib/PoseLib.git poselib_src
cd poselib_src
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/../../eigen3_install"
cmake --build . --config Release
cd ../..

# Build with CMake
mkdir build && cd build
cmake .. -DOpenCV_DIR="C:/tools/opencv/build" -DCMAKE_PREFIX_PATH="$PWD/../eigen3_install"
cmake --build . --config Release

# Binary will be at: build/Release/vo_submission.exe
```

### Alternative: Direct Compilation (Linux/macOS)

```bash
g++ -O3 -std=c++17 -o vo_submission src/vo_submission.cpp \
    -I poselib_src -I poselib_src/build/generated_headers \
    poselib_src/build/PoseLib/libPoseLib.a \
    $(pkg-config --cflags --libs opencv4 eigen3)
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
| Auto focal (no flags) | ~0.55s | Default, no calibration needed |
| Known focal, scale=0.5 | ~0.35s | When calibration known |
| Known focal, scale=1.0 | ~0.70s | High precision |
| Known focal, scale=0.25 | ~0.15s | Real-time embedded |

## Acknowledgments

This project was developed with assistance from [Claude Code](https://claude.ai/code) AI.
