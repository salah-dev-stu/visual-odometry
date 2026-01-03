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

#### Run
```powershell
vo_submission.exe image1.jpg image2.jpg
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
