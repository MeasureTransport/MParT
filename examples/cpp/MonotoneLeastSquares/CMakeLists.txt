cmake_minimum_required (VERSION 3.13)

project(MonotoneLeastSquares)

set(CMAKE_CXX_STANDARD 17)

find_package(MParT REQUIRED)
message(STATUS "MPART_FOUND = ${MParT_FOUND}")

add_executable(MonotoneLeastSquares MonotoneLeastSquares.cpp)
target_link_libraries(MonotoneLeastSquares MParT::mpart Kokkos::kokkos Eigen3::Eigen)
