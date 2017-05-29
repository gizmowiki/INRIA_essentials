get_filename_component(NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(example-sup-${NAME})

## #################################################################
## Sources
## #################################################################

set(${PROJECT_NAME}_HEADERS )

set(${PROJECT_NAME}_HEADERS_MOC )

set(${PROJECT_NAME}_SOURCES main.cpp)

##
find_package(OpenCV 2.3 REQUIRED core imgproc highgui)

## #################################################################
## Build rules
## #################################################################

qt4_wrap_cpp(${PROJECT_NAME}_SOURCES_MOC ${${PROJECT_NAME}_HEADERS_MOC})

add_executable(${PROJECT_NAME}
  ${${PROJECT_NAME}_HEADERS}
  ${${PROJECT_NAME}_SOURCES_MOC}
  ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME}
  ${QT_LIBRARIES}
  dtkLog
  dtkCore
  supCore
  ${OpenCV_LIBS})

if( NOT APPLE )
target_link_libraries(${PROJECT_NAME}
  rt)
endif( NOT APPLE )


## #################################################################
## Installation
## #################################################################

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin)
