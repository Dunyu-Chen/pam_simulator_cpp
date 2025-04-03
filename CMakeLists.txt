cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pam_simulator_cpp)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

set(Torch_DIR /opt/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test_exec src/test_exec.cpp)
target_link_libraries(test_exec PUBLIC lib_simulator lib_deepLearning)
ament_target_dependencies(test_exec PUBLIC rclcpp std_msgs)

install(
  TARGETS test_exec
  DESTINATION lib/${PROJECT_NAME})

add_library(lib_simulator INTERFACE
  include/pam_simulator_cpp/lib_simulator.h
)
target_include_directories(lib_simulator INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
target_link_libraries(lib_simulator INTERFACE
    Eigen3::Eigen "${TORCH_LIBRARIES}")

add_library(lib_deepLearning INTERFACE
        include/pam_simulator_cpp/lib_deepLearning.h
)
target_include_directories(lib_deepLearning INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
target_link_libraries(lib_deepLearning INTERFACE
        Eigen3::Eigen "${TORCH_LIBRARIES}")

# ament_target_dependencies(my_library INTERFACE rclcpp)
install(
    DIRECTORY include/
    DESTINATION include/${PROJECT_NAME}
)
install(
    TARGETS lib_simulator lib_deepLearning
    EXPORT export_${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
# export targets
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies()

ament_package()
