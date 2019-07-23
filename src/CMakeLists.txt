add_definitions(-fPIC)
# Build main library.
set(FLAME_SRCS
        flame/flame.h
        flame/flame.cc

        flame/params.h
        flame/types.h

        flame/external/triangle/triangle.h
        flame/external/triangle/triangle.cpp

        flame/optimizers/nltgv2_l1_graph_regularizer.h
        flame/optimizers/nltgv2_l1_graph_regularizer.cc

        flame/stereo/epipolar_geometry.h
        flame/stereo/inverse_depth_filter.h
        flame/stereo/inverse_depth_filter.cc
        flame/stereo/inverse_depth_meas_model.h
        flame/stereo/inverse_depth_meas_model.cc
        flame/stereo/line_stereo.h

        flame/utils/assert.h
        flame/utils/delaunay.h
        flame/utils/delaunay.cc
        flame/utils/frame.h
        flame/utils/frame.cc
        flame/utils/image_utils.h
        flame/utils/image_utils.cc
        flame/utils/keyframe_selector.h
        flame/utils/keyframe_selector.cc
        flame/utils/load_tracker.h
        flame/utils/pyramids.h
        flame/utils/pyramids.cc
        flame/utils/rasterization.h
        flame/utils/rasterization.cc
        flame/utils/stats_tracker.h
        flame/utils/triangulator.h
        flame/utils/visualization.h)

add_library(${appname}-core SHARED ${FLAME_SRCS})

target_link_libraries(${appname}-core
        ${common_libraries}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES})
set_target_properties(${appname}-core PROPERTIES LINKER_LANGUAGE CXX COMPILE_FLAGS "-fPIC")

install(FILES
        ${PROJECT_BINARY_DIR}/lib/cmake/${appname}-core/${appname}-coreConfig.cmake
        ${PROJECT_BINARY_DIR}/lib/cmake/${appname}-core/${appname}-coreConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/${appname}-core)

install(TARGETS ${appname}-core
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# build slambench flame library
add_library(${appname} SHARED
            flame/flame_library.cpp)

set_target_properties(${appname} PROPERTIES COMPILE_FLAGS "-fPIC")

target_link_libraries(${appname}
                      ${appname}-core
                      ${OpenCV_LIBS}
                      ${PCL_LIBRARIES})

target_include_directories(${appname} PUBLIC
        ${EIGEN3_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        )

generate_slam_library(${appname})

message("OpenCV LIBS: " ${OpenCV_LIBS})
message("OpenCV Includes: " ${OpenCV_INCLUDE_DIRS})
message("PCL INCUDES: " ${PCL_INCLUDE_DIRS})
#target_link_libraries(${appname}-library ${OpenCV_LIBS} ${appname})
#target_include_directories(${appname}-library PRIVATE ${OpenCV_INCLUsDE_DIRS})