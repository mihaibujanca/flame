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
        flame/utils/visualization.h
        flame/flame_library.cpp)

add_library(${appname} SHARED ${FLAME_SRCS})

target_link_libraries(${appname}
        ${common_libraries}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

install(FILES
        ${PROJECT_BINARY_DIR}/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake
        ${PROJECT_BINARY_DIR}/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME})

install(TARGETS ${appname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

generate_slam_library(${appname})
message("OpenCV LIBS: " ${OpenCV_LIBS})
message("OpenCV Includes: " ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${appname}-library ${OpenCV_LIBS})
target_include_directories(${appname}-library PRIVATE ${OpenCV_INCLUDE_DIRS})