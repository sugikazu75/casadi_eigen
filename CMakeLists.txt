cmake_minimum_required(VERSION 3.0.2)
project(casadi_eigen)

add_compile_options(-std=c++17)

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS)

find_package(casadi REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES casadi_eigen
  )

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${NLOPT_INCLUDE_DIRS}
  )

add_library(casadi_eigen src/CasadiEigen.cpp)
target_link_libraries(casadi_eigen
  casadi
  ${EIGEN3_LIBRARIES}
  )

add_executable(casadi_sample1 src/casadi_sample1.cpp)
target_link_libraries(casadi_sample1
  casadi
  ${catkin_LIBRARIES}
  )

add_executable(casadi_sample2 src/casadi_sample2.cpp)
target_link_libraries(casadi_sample2
  casadi
  ${catkin_LIBRARIES}
  )

add_executable(casadi_sample3 src/casadi_sample3.cpp)
target_link_libraries(casadi_sample3
  casadi
  casadi_eigen
  ${catkin_LIBRARIES}
  )

add_executable(casadi_test src/casadi_test.cpp)
target_link_libraries(casadi_test
  casadi
  ${catkin_LIBRARIES}
  )
