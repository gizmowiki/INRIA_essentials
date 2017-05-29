project(SSD1IMG)

## #################################################################
## Sources
## #################################################################

set(${PROJECT_NAME}_HEADERS people_detector.h)

set(${PROJECT_NAME}_SOURCES main.cpp people_detector.cpp)

##
find_package(OpenCV REQUIRED) #core imgproc highgui nonfree

## #################################################################
## Build rules
## #################################################################

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

add_definitions(-std=c++11)

##############################################################

include_directories(
    ${OpenCV_INCLUDE_DIRS})

##############################################################

##################################################################

set(Caffe_DIR "/user/hunnguye/home/project/deep_learning/experiment/caffe/build")
find_package(Caffe REQUIRED)
INCLUDE_DIRECTORIES(${Caffe_INCLUDE_DIRS})

##################################################################

add_executable(${PROJECT_NAME}
  ${${PROJECT_NAME}_HEADERS}
  ${${PROJECT_NAME}_SOURCES_MOC}
  ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Caffe_LIBRARIES}
)

## #################################################################
## Installation
## #################################################################

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin)
