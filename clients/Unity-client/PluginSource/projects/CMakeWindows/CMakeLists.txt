cmake_minimum_required(VERSION 3.18)
project(RenderingPlugin C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
set(INCLUDEDIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
set(DEPSDIR ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies)

set(SRCS
    ${SRCDIR}/RenderingPlugin.cpp
    ${SRCDIR}/RenderAPI.cpp
    ${SRCDIR}/RenderAPI_OpenGLCoreES.cpp
    ${DEPSDIR}/gl3w/gl3w.c
)

# Platform defines
if(WIN32)
    add_definitions(-DUNITY_WIN=1)
    list(APPEND SRCS
        ${SRCDIR}/RenderAPI_D3D11.cpp
        ${SRCDIR}/RenderAPI_D3D12.cpp
    )
else()
    add_definitions(-DUNITY_LINUX=1)
endif()

add_definitions(
    -DSUPPORT_OPENGL_UNIFIED=1
    -DSUPPORT_OPENGL_CORE=1
    -DSUPPORT_VULKAN=0
)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find OpenGL
if(WIN32)
    find_package(OpenGL REQUIRED)
else()
    set(OpenGL_GL_PREFERENCE GLVND)
    find_package(OpenGL REQUIRED)
endif()

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SRCS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${INCLUDEDIR}
    ${SRCDIR}
    ${DEPSDIR}
    ${CUDAToolkit_INCLUDE_DIR}
)

# Platform-specific settings
if(WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE /O2 /W3)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        OpenGL::GL
        CUDA::cudart
        ws2_32
        d3d11
        dxgi
    )
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME "RenderingPlugin"
        SUFFIX ".dll"
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -O2 -fPIC)
    target_link_options(${PROJECT_NAME} PRIVATE -rdynamic)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        OpenGL::GL
        CUDA::cudart
    )
    set_target_properties(${PROJECT_NAME} PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "RenderingPlugin"
        SUFFIX ".so"
    )
endif()
