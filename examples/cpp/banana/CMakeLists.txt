cmake_minimum_required (VERSION 3.13)

project(Banana)

set(CMAKE_CXX_STANDARD 17)

find_package(MParT REQUIRED)

add_executable(Banana EvaluateBanana.cpp)
target_link_libraries(Banana MParT::mpart Kokkos::kokkos Eigen3::Eigen)
