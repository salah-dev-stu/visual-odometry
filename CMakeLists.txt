cmake_minimum_required(VERSION 3.10)
project(vo_submission)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# PoseLib
set(POSELIB_DIR ${CMAKE_SOURCE_DIR}/poselib_src)
set(POSELIB_INCLUDE ${POSELIB_DIR})
if(WIN32)
    set(POSELIB_LIB ${POSELIB_DIR}/build/PoseLib/Release/PoseLib.lib)
else()
    set(POSELIB_LIB ${POSELIB_DIR}/build/PoseLib/libPoseLib.a)
endif()

add_executable(vo_submission src/vo_submission.cpp)
target_include_directories(vo_submission PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${POSELIB_INCLUDE}
    ${EIGEN3_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(vo_submission ${OpenCV_LIBS} ${POSELIB_LIB})

# ─── ONNX Runtime (optional, for vo_neural) ──────────────────────────────────
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime)
if(EXISTS ${ONNXRUNTIME_DIR}/include)
    message(STATUS "ONNX Runtime found at: ${ONNXRUNTIME_DIR}")
    set(ONNXRUNTIME_INCLUDE ${ONNXRUNTIME_DIR}/include)

    # Find the shared library
    file(GLOB ONNXRUNTIME_LIB_FILES
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so*"
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib*"
    )
    if(ONNXRUNTIME_LIB_FILES)
        list(GET ONNXRUNTIME_LIB_FILES 0 ONNXRUNTIME_LIB)
    else()
        set(ONNXRUNTIME_LIB "")
    endif()

    if(ONNXRUNTIME_LIB)
        message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")

        # Check for CUDA provider (GPU build)
        if(EXISTS "${ONNXRUNTIME_DIR}/lib/libonnxruntime_providers_cuda.so")
            set(ORT_HAS_CUDA TRUE)
            message(STATUS "ONNX Runtime CUDA provider found — GPU acceleration enabled")
        else()
            set(ORT_HAS_CUDA FALSE)
            message(STATUS "ONNX Runtime CUDA provider not found — CPU only")
        endif()

        add_executable(vo_neural src/vo_neural.cpp)
        target_include_directories(vo_neural PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${POSELIB_INCLUDE}
            ${EIGEN3_INCLUDE_DIR}
            ${EIGEN3_INCLUDE_DIRS}
            ${ONNXRUNTIME_INCLUDE}
        )
        target_link_libraries(vo_neural ${OpenCV_LIBS} ${POSELIB_LIB} ${ONNXRUNTIME_LIB})

        if(ORT_HAS_CUDA)
            target_compile_definitions(vo_neural PRIVATE USE_CUDA)
        endif()

        # Set RPATH so the binary finds libonnxruntime at runtime
        set_target_properties(vo_neural PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
        )
    else()
        message(WARNING "ONNX Runtime lib not found in ${ONNXRUNTIME_DIR}/lib — vo_neural will NOT be built")
    endif()
else()
    message(STATUS "ONNX Runtime not found at ${ONNXRUNTIME_DIR} — vo_neural will NOT be built")
    message(STATUS "  Run: bash scripts/download_models.sh")
endif()

# Trajectory Viewer with Pangolin
find_package(Pangolin QUIET)
if(Pangolin_FOUND)
    add_executable(trajectory_viewer src/trajectory_viewer.cpp)
    target_link_libraries(trajectory_viewer ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
    target_include_directories(trajectory_viewer PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()
