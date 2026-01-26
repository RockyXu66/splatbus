cmake_minimum_required(VERSION 3.10)
project(RenderingPlugin C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME RenderingPlugin)

# Source directory
set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
set(INCLUDEDIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
set(DEPSDIR ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies)

# Source files
set(SRCS

    # ${SRCDIR}/include/CudaKernels.h

    ${SRCDIR}/RenderingPlugin.cpp
    ${SRCDIR}/RenderAPI.cpp
    ${SRCDIR}/RenderAPI_OpenGLCoreES.cpp
    # ${SRCDIR}/RenderAPI_Vulkan.cpp
    ${DEPSDIR}/gl3w/gl3w.c

    # ${SRCDIR}/CudaKernels.cu
)

# Unity defines
add_definitions(
    -DSUPPORT_OPENGL_UNIFIED=1
    # -DSUPPORT_OPENGL_CORE=1
    -DSUPPORT_VULKAN=0
    -DUNITY_LINUX=1
)


set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
# set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")


# Set OpenGL preference to use GLVND (modern approach)
set(OpenGL_GL_PREFERENCE GLVND)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#     set(CMAKE_PREFIX_PATH "/home/yixu/Tools/libtorch-shared-with-deps-2.9.0+cu126")
# else()
#     set(CMAKE_PREFIX_PATH "/home/yixu/Tools/libtorch-shared-with-deps-2.9.0+cu126")
# endif()

# # Find Torch
# find_package(Torch REQUIRED)

include_directories(/usr/local/include)
include_directories(/usr/include/eigen3)
link_directories(/usr/local/lib)

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SRCS})


# ==================== DIAGNOSTIC BLOCK ====================
# message(STATUS "--- LibTorch Diagnostics ---")
# message(STATUS "TORCH_FOUND: ${TORCH_FOUND}")
# message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
# message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
# message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")

message(STATUS "--- End Diagnostics ---")
# ===========================================================================



# list(APPEND UNITYNI_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}")

list(APPEND UNITYNI_INCLUDE_DIRECTORIES "${INCLUDEDIR}")
list(APPEND UNITYNI_INCLUDE_DIRECTORIES "${SRCDIR}")
list(APPEND UNITYNI_INCLUDE_DIRECTORIES "${DEPSDIR}")
list(APPEND UNITYNI_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIR}")

# Include CUDA headers and local headers
target_include_directories(${PROJECT_NAME} PUBLIC ${UNITYNI_INCLUDE_DIRECTORIES})

# Set compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -O2
    -fPIC
)

# Set linker flags
target_link_options(${PROJECT_NAME} PRIVATE
    -rdynamic
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    CUDA::cudart
    # ${TORCH_LIBRARIES}
)

# Set output name
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "${PROJECT_NAME}"
    SUFFIX ".so"
)

