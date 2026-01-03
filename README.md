# Visual Odometry - 2-Frame Pose Estimation

Estimates relative camera pose (6DOF) between two images without camera calibration.

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
g++ -O3 -o vo_submission src/vo_submission.cpp $(pkg-config --cflags --libs opencv4)
```

### macOS
```bash
# Install OpenCV via Homebrew
brew install opencv

# Build
g++ -O3 -std=c++17 -o vo_submission src/vo_submission.cpp $(pkg-config --cflags --libs opencv4)
```

### Windows

#### Option 1: vcpkg
```powershell
# Install vcpkg and OpenCV
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install opencv4:x64-windows

# Build (from project directory)
cl /EHsc /O2 /std:c++17 src\vo_submission.cpp /I vcpkg\installed\x64-windows\include /link /LIBPATH:vcpkg\installed\x64-windows\lib opencv_world4*.lib
```

#### Option 2: Pre-built OpenCV
1. Download OpenCV from https://opencv.org/releases/
2. Extract and set `OPENCV_DIR` environment variable
3. Build:
```powershell
cl /EHsc /O2 /std:c++17 src\vo_submission.cpp /I %OPENCV_DIR%\include /link /LIBPATH:%OPENCV_DIR%\x64\vc16\lib opencv_world4*.lib
```

## Usage
```bash
./vo_submission image1.jpg image2.jpg
```

## Example
```bash
./vo_submission frame001.jpg frame002.jpg
# Output: 0.523415 -0.128743 0.892156 0.125634 0.987234 -0.098123
```
