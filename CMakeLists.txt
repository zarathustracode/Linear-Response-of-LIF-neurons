cmake_minimum_required(VERSION 3.10.2)
project(LIF0 VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#find_package (Python3 COMPONENTS Interpreter Development)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Boost 1.7 REQUIRED python numpy)
find_package (Eigen3 REQUIRED)

message("-- Python3 inlcudes ${PYTHON_INCLUDE_DIRS}")

message("-- Python3 libraries ${PYTHON_LIBRARIES}")

message("-- Boost inlcudes ${Boost_INCLUDE_DIR}")

message("-- Boost libraries ${Boost_LIBRARIES}")


IF( NOT EIGEN3_INCLUDE_DIR )
    message("There is not Eigen3. Setting it manualy")
    SET( EIGEN3_INCLUDE_DIR "/usr/include/eigen3" )
ENDIF()

message("-- Eigen3 include: ${EIGEN3_INCLUDE_DIR}")

include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
link_directories("/usr/local/lib") # I had to do this on ubuntu for some reason

add_library(LIF0 SHARED LIF0.cpp)
add_executable(main main.cpp)

target_link_libraries(LIF0 ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

target_link_libraries(main LIF0)

# Tweaks the name of the library to match what Python expects
set_target_properties(LIF0 PROPERTIES SUFFIX .so)
set_target_properties(LIF0 PROPERTIES PREFIX "")

#Run in debug mode cmake -DCMAKE_BUILD_TYPE=Debug .
#Run in release mode cmake -DCMAKE_BUILD_TYPE=Release .
