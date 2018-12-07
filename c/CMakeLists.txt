cmake_minimum_required(VERSION 2.6)
project(gomoku)

SET( CMAKE_BUILD_TYPE Debug  )

IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()

# find_package(PythonLibs REQUIRED)
SET(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython3.6m.so)
SET(NUMPY_LIBRARIES /home/biwefc/.local/lib/python3.6/site-packages/numpy/core/lib/libnpymath.a)

include_directories(${PROJECT_SOURCE_DIR}/include/ /usr/include/python3.6m /home/biwefc/.local/lib/python3.6/site-packages/numpy/core/include/)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(gomokuer.exe ${SRC})
TARGET_LINK_LIBRARIES(gomokuer.exe ${PYTHON_LIBRARIES} ${NUMPY_LIBRARIES})

